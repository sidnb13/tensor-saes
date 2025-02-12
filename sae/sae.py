import json
import math
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributed._tensor import DTensor, Replicate

from sae.config import SaeConfig

from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone()) if decoder else None

        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

        if self.cfg.post_act_bias:
            self.post_act_bias = nn.Parameter(
                torch.zeros(self.num_latents, dtype=dtype, device=device)
            )
        else:
            self.post_act_bias = None

        self.tp_mesh = None

    def handle_dec_bias(self, x: torch.Tensor, op="add") -> torch.Tensor | DTensor:
        if isinstance(self.b_dec, DTensor):
            x = DTensor.from_local(x, self.tp_mesh, placements=[Replicate()])

        return x + self.b_dec if op == "add" else x - self.b_dec

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | list[tuple[str, ...]] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            if isinstance(layers[0], tuple):
                layers = ["_".join(layer) for layer in layers]
            return {
                layer: Sae.load_from_disk(
                    repo_path / layer,
                    device=device,
                    decoder=decoder,
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        step: int | None = None,
    ) -> "Sae":
        # TODO: sidnb13 collect sharded weights into unified SAE on loading.

        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig(**cfg_dict)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        sae_weights = load_file(
            str(
                path
                / ("sae.safetensors" if step is None else f"sae-{step}.safetensors")
            ),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
        )
        sae.encoder.weight = sae_weights["encoder.weight"]
        sae.encoder.bias = sae_weights["encoder.bias"]
        sae.W_dec = sae_weights["decoder.weight"]
        sae.b_dec = sae_weights["decoder.bias"]
        return sae

    def save_config(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = self.handle_dec_bias(x.to(self.dtype), op="sub")
        out = self.encoder(sae_in)

        return nn.functional.relu(out) if not self.cfg.signed else out

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        if self.cfg.signed:
            _, top_indices = latents.abs().topk(self.cfg.k, sorted=False)
            top_acts = latents.gather(dim=-1, index=top_indices)

            return EncoderOutput(top_acts, top_indices)

        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        if self.tp_mesh is not None:
            wdec = DTensor.to_local(self.W_dec).mT
        else:
            wdec = self.W_dec.mT

        y = decoder_impl(top_indices, top_acts.to(self.dtype), wdec)
        return self.handle_dec_bias(y, op="add")

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ForwardOutput:
        pre_acts = self.pre_acts(x)
        top_acts, top_indices = self.select_topk(pre_acts)

        if self.post_act_bias is not None:
            top_acts = top_acts + self.post_act_bias.expand(x.shape[0], -1).gather(
                1, top_indices
            )

        # Decode and compute residual
        sae_out = self.decode(top_acts, top_indices)

        if self.tp_mesh is not None:
            sae_out = DTensor.to_local(sae_out)

        e = sae_out - x

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum(0)
            auxk_loss = scale * torch.mean(auxk_loss / total_variance)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum(0)
        fvu = torch.mean(l2_loss / total_variance)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )

    @torch.no_grad()
    def scale_encoder_k(self):
        scale = 1 / math.sqrt(self.cfg.k)
        self.encoder.weight.data *= scale
        self.encoder.bias.data *= scale

    @torch.no_grad()
    def scale_encoder_fvu(
        self, mean_total_variance: Tensor, mean_output_variance: Tensor
    ):
        scale = self.cfg.scale_encoder_fvu_batch * torch.sqrt(
            torch.mean(mean_total_variance) / torch.mean(mean_output_variance)
        )
        self.encoder.weight.data *= scale
        self.encoder.bias.data *= scale

    @torch.no_grad()
    def compute_in_out_var(self, all_hiddens: Tensor, chunk_size):
        total_variance = (all_hiddens - all_hiddens.mean(0)).pow(2).sum(0)
        # compute output variance
        output_variance = torch.zeros(
            *all_hiddens.shape[1:], device=all_hiddens.device, dtype=all_hiddens.dtype
        )
        for chunk in all_hiddens.chunk(chunk_size):
            sae_out = self(chunk).sae_out
            output_variance += (sae_out - sae_out.mean(0)).pow(2).sum(0)

        return torch.mean(total_variance), torch.mean(output_variance)

    @torch.no_grad()
    def scale_encoder_fvu_batch(self, all_hiddens: Tensor, chunk_size):
        # (batch, seq, hidden * num_layers)
        total_variance = (all_hiddens - all_hiddens.mean(0)).pow(2).sum(0)
        # compute output variance
        output_variance = torch.zeros(
            *all_hiddens.shape[1:], device=all_hiddens.device, dtype=all_hiddens.dtype
        )
        for chunk in all_hiddens.chunk(chunk_size):
            sae_out = self(chunk).sae_out
            output_variance += (sae_out - sae_out.mean(0)).pow(2).sum(0)
        # scale encoder weights
        mean_tot_var, mean_out_var = torch.mean(total_variance), torch.mean(output_variance)
        scale = self.cfg.scale_encoder_fvu_batch * torch.sqrt(
            mean_tot_var / mean_out_var
        )
        self.encoder.weight.data *= scale
        self.encoder.bias.data *= scale

        return mean_tot_var, mean_out_var

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
