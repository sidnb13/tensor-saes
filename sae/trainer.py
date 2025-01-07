import os
from collections import defaultdict
from collections.abc import Sized
from dataclasses import asdict
from fnmatch import fnmatchcase

import torch
import torch.distributed as dist
from natsort import natsorted
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed._tensor import DTensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from .config import TrainConfig
from .logger import get_logger
from .sae import Sae
from .utils import (
    configure_tp_model,
    geometric_median,
    get_layer_list,
    log_parameter_norms,
    resolve_widths,
    resolve_widths_rangewise,
)

logger = get_logger(__name__)


class SaeTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        train_dataset: Dataset,
        test_dataset: Dataset,
        model: PreTrainedModel,
        *args,
        **kwargs,
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for name, _ in model.named_modules():
                if any(fnmatchcase(name, pat) for pat in cfg.hookpoints):
                    raw_hookpoints.append(name)

            # Natural sort to impose a consistent order
            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N, cfg.layer_stride))

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.distribute_modules()

        N = len(cfg.hookpoints)
        assert isinstance(train_dataset, Sized)
        num_examples = len(train_dataset)

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}",
            )

        self.model = model
        self.saes = {
            hook: Sae(input_widths[hook], cfg.sae, device)
            for hook in self.local_hookpoints()
        }

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5,
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        logger.info(
            f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}",
        )

        try:
            from bitsandbytes.optim import Adam8bit as Adam  # type: ignore  # noqa: I001

            logger.info("Using 8-bit Adam from bitsandbytes")
        except ImportError:
            from torch.optim import Adam

            logger.info("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
            logger.info("Run `pip install bitsandbytes` for less memory usage.")

        logger.info(f"Using optimizer: {cfg.optimizer}")

        if cfg.optimizer == "adam":
            self.optimizer = Adam(pgs)
        elif cfg.optimizer == "adam_zero":
            self.optimizer = ZeroRedundancyOptimizer(pgs, Adam)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            cfg.lr_warmup_steps,
            num_examples // (cfg.batch_size * cfg.micro_acc_steps),
        )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    group=self.cfg.wandb_group,
                    save_code=True,
                )
            except ImportError:
                logger.info("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Number of SAE parameters: {num_sae_params:_}")
        logger.info(f"Number of model parameters: {num_model_params:_}")

        device = self.model.device
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        pbar = tqdm(dl, desc="Training", disable=not rank_zero)

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_loss = defaultdict(float)
        avg_fvu = defaultdict(float)

        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {
            name: self.model.get_submodule(name) for name in self.cfg.hookpoints
        }
        module_to_name = {v: k for k, v in name_to_module.items()}

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            name = module_to_name[module]
            hidden_dict[name] = outputs.flatten(0, 1)

        for i, batch in enumerate(pbar):
            hidden_dict.clear()

            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]
            try:
                with torch.no_grad():
                    self.model(batch["input_ids"].to(device))
            finally:
                for handle in handles:
                    handle.remove()

            if self.cfg.distribute_modules:
                hidden_dict = self.scatter_hiddens(hidden_dict)

            grad_norms = {}

            for name, hiddens in hidden_dict.items():
                # normalize hiddens to have unit norm
                if self.cfg.normalize_hiddens:
                    hiddens = hiddens / hiddens.norm(dim=-1, keepdim=True)

                raw = self.saes[name]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                if i == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    raw.b_dec.data = median.to(raw.dtype)

                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                    if raw.cfg.scale_encoder_fvu_global:
                        logger.info(
                            "Computing global mean and variance for FVU scaling",
                        )
                        total_variance = torch.zeros(
                            hiddens.shape[-1],
                            device=self.model.device,
                            dtype=torch.float32,
                        )
                        output_variance = torch.zeros(
                            hiddens.shape[-1],
                            device=self.model.device,
                            dtype=torch.float32,
                        )
                        test_loader = DataLoader(
                            self.test_dataset, batch_size=self.cfg.batch_size,
                        )

                        hidden_sum = torch.zeros_like(total_variance)
                        total_tokens = 0

                        for batch in test_loader:
                            hidden_dict.clear()
                            handles = [
                                mod.register_forward_hook(hook)
                                for mod in name_to_module.values()
                            ]
                            try:
                                with torch.no_grad():
                                    self.model(batch["input_ids"].to(device))
                            finally:
                                for handle in handles:
                                    handle.remove()

                            batch_hiddens = hidden_dict[name]
                            all_hiddens = self.maybe_all_cat(batch_hiddens)
                            hidden_sum += all_hiddens.sum(0)
                            total_tokens += all_hiddens.shape[0]

                        global_mean = hidden_sum / total_tokens

                        for batch in test_loader:
                            hidden_dict.clear()
                            handles = [
                                mod.register_forward_hook(hook)
                                for mod in name_to_module.values()
                            ]
                            try:
                                with torch.no_grad():
                                    self.model(batch["input_ids"].to(device))
                            finally:
                                for handle in handles:
                                    handle.remove()

                            batch_hiddens = hidden_dict[name]
                            all_hiddens = self.maybe_all_cat(batch_hiddens)

                            total_variance += ((all_hiddens - global_mean).pow(2)).sum(
                                0,
                            )

                            for chunk in all_hiddens.chunk(self.cfg.micro_acc_steps):
                                with torch.no_grad():
                                    reconstructed = raw(chunk).sae_out
                                    output_variance += (
                                        (reconstructed - chunk).pow(2)
                                    ).sum(0)

                        total_variance /= total_tokens
                        output_variance /= total_tokens

                        raw.scale_encoder_fvu(total_variance, output_variance)

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                if raw.cfg.scale_encoder_k:
                    raw.scale_encoder_k()

                all_hiddens = self.maybe_all_cat(hiddens)
                if raw.cfg.scale_encoder_fvu_batch:
                    in_var, out_var = raw.scale_encoder_fvu_batch(all_hiddens, self.cfg.micro_acc_steps)
                else:
                    in_var, out_var = raw.compute_in_out_var(all_hiddens, self.cfg.micro_acc_steps)

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[name]

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            num_tokens_since_fired[name]
                            > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[name] += float(
                        self.maybe_all_reduce(out.fvu.detach()) / denom,
                    )
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom,
                        )

                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                    loss.div(acc_steps).backward()

                    avg_loss[name] += float(
                        self.maybe_all_reduce(loss.detach()) / denom,
                    )

                    # Update the did_fire mask
                    did_fire[name][out.latent_indices.flatten()] = True
                    self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                grad_norms[name] = torch.nn.utils.clip_grad_norm_(
                    raw.parameters(), 1.0,
                ).item()

            # Check if we need to actually do a training step
            step, substep = divmod(i + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                info = {}

                for names in self.saes:
                    mask = (
                        num_tokens_since_fired[names] > self.cfg.dead_feature_threshold
                    )

                    names_str = "_".join(names)

                    info.update(
                        {
                            f"fvu/{names_str}": avg_fvu[names],
                            f"dead_pct/{names_str}": mask.mean(
                                dtype=torch.float32,
                            ).item(),
                            f"loss/{names_str}": avg_loss[names],
                            f"lr/{names_str}": self.optimizer.param_groups[-1]["lr"],
                            f"grad_norm/{names_str}": grad_norms[names],
                            "step": step,
                        },
                    )

                    if out_var is not None and in_var is not None:
                        info[f"in_var/{names_str}"] = in_var.cpu().item()
                        info[f"out_var/{names_str}"] = out_var.cpu().item()

                    if self.cfg.auxk_alpha > 0:
                        info[f"auxk/{names}"] = avg_auxk_loss[names]

                    # Log parameter norms
                    log_parameter_norms(self.saes[names], names_str, info)

                if (step + 1) % min(
                    self.cfg.stdout_log_frequency, self.cfg.wandb_log_frequency,
                ) == 0 and rank_zero:
                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_loss.clear()

                if (step + 1) % self.cfg.stdout_log_frequency == 0 and rank_zero:
                    logger.info(info)

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save(step)

        if rank_zero and self.cfg.log_to_wandb:
            wandb.finish()

        self.save(step)
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return (
            self.module_plan[dist.get_rank()]
            if self.module_plan
            else self.cfg.hookpoints
        )

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            logger.info(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            logger.info(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self, step: int):
        """Save to disk."""
        if (
            self.cfg.distribute_modules
            or not dist.is_initialized()
            or dist.get_rank() == 0
        ):
            logger.info("Saving checkpoint")

            for hook, sae in self.saes.items():
                assert isinstance(sae, Sae)

                hook_name = "_".join(hook)

                path = self.cfg.run_name or "checkpoints"
                full_path = f"{self.cfg.root_path}/{path}/{hook_name}"

                # Ensure the directory exists
                os.makedirs(full_path, exist_ok=True)

                # Save the state dict instead of pickling the entire object
                sae_state = {
                    "encoder": sae.encoder.state_dict(),
                    "decoder": DTensor.to_local(sae.W_dec)
                    if isinstance(sae.W_dec, DTensor)
                    else sae.W_dec,
                    "b_dec": DTensor.to_local(sae.b_dec)
                    if isinstance(sae.b_dec, DTensor)
                    else sae.b_dec,
                }

                torch.save(sae_state, f"{full_path}/sae_state.pt")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()


class SaeLayerRangeTrainer(SaeTrainer):
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: Dataset,
        model: PreTrainedModel,
        rank,
        world_size,
    ):
        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Replace wildcard patterns
            raw_hookpoints = []
            for segment in cfg.hookpoints:
                for name, _ in model.named_modules():
                    if any(fnmatchcase(name, pat) for pat in segment):
                        raw_hookpoints.append(name)

                # Natural sort to impose a consistent order
                sorted_hookpoints = natsorted(raw_hookpoints)
                raw_hookpoints.append(tuple(sorted_hookpoints))

            cfg.hookpoints = raw_hookpoints
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = [tuple(range(N))]
            else:
                cfg.layers = [sorted(lyr) for lyr in cfg.layers]

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            raw_hookpoints = []

            for segment_layers in cfg.layers:
                segment_hookpoints = [f"{layers_name}.{i}" for i in segment_layers]
                raw_hookpoints.append(tuple(segment_hookpoints))

            cfg.hookpoints = raw_hookpoints

        self.cfg = cfg
        self.dataset = dataset
        self.distribute_modules()

        assert isinstance(dataset, Sized)
        num_examples = len(dataset)

        device = model.device
        input_widths = resolve_widths_rangewise(model, cfg.hookpoints)
        unique_widths = set(input_widths.values())

        if cfg.distribute_modules and len(unique_widths) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_widths}",
            )

        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.saes = {
            hook_segment: Sae(input_widths[hook_segment], cfg.sae, device)
            for hook_segment in self.local_hookpoints()
        }

        if self.cfg.tp:
            logger.info("Configuring tensor parallelism")
            self.saes = {
                hook_segment: configure_tp_model(sae, self.world_size)
                for hook_segment, sae in self.saes.items()
            }

        pgs = [
            {
                "params": sae.parameters(),
                # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
                "lr": cfg.lr or 2e-4 / (sae.num_latents / (2**14)) ** 0.5,
            }
            for sae in self.saes.values()
        ]
        # Dedup the learning rates we're using, sort them, round to 2 decimal places
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]
        logger.info(
            f"Learning rates: {lrs}" if len(lrs) > 1 else f"Learning rate: {lrs[0]}",
        )

        if "8bit" in cfg.optimizer:
            try:
                from bitsandbytes.optim import Adam8bit as Adam  # type: ignore  # noqa: I001
                from bitsandbytes.optim import AdamW8bit as AdamW  # type: ignore

                logger.info(f"Using 8-bit {cfg.optimizer} from bitsandbytes")
            except ImportError:
                from torch.optim import Adam, AdamW

                logger.info(
                    "bitsandbytes 8-bit Adam not available, using torch.optim.Adam",
                )
                logger.info("Run `pip install bitsandbytes` for less memory usage.")
                logger.info(f"Using optimizer: {cfg.optimizer}")
        else:
            from torch.optim import Adam, AdamW

        if "adam" in cfg.optimizer:
            self.optimizer = Adam(pgs)
        elif "adamw" in cfg.optimizer:
            self.optimizer = AdamW(pgs)
        elif cfg.optimizer == "adam_zero":
            self.optimizer = ZeroRedundancyOptimizer(pgs, Adam)
        elif cfg.optimizer == "adamw_zero":
            self.optimizer = ZeroRedundancyOptimizer(pgs, AdamW)

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size,
        )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = self.cfg.ddp and dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    group=self.cfg.wandb_group,
                    save_code=True,
                )
            except ImportError:
                logger.info("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Number of SAE parameters: {num_sae_params:_}")
        logger.info(f"Number of model parameters: {num_model_params:_}")

        device = self.model.device
        dl = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        pbar = tqdm(dl, desc="Training", disable=not rank_zero)

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        avg_loss = defaultdict(float)
        avg_fvu = defaultdict(float)

        hidden_dict = defaultdict(list)
        name_to_module_list = {
            hookpoints: tuple([self.model.get_submodule(name) for name in hookpoints])
            for hookpoints in self.cfg.hookpoints
        }

        def hook(module: nn.Module, _, outputs):
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            key = next(
                (
                    names
                    for names, mods in name_to_module_list.items()
                    if module in mods
                ),
                None,
            )
            hidden_dict[key].append(outputs.flatten(0, 1))

        for i, batch in enumerate(pbar):
            hidden_dict = defaultdict(list)

            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations
            nested_handles = [
                [mod.register_forward_hook(hook) for mod in mods]
                for mods in name_to_module_list.values()
            ]
            try:
                with torch.no_grad():
                    self.model(batch["input_ids"].to(device))
                # concatenate outputs in hidden_dict
                hidden_dict = {
                    key: torch.cat(outputs, dim=-1)
                    for key, outputs in hidden_dict.items()
                }
            finally:
                for handles in nested_handles:
                    for h in handles:
                        h.remove()

            if self.cfg.distribute_modules:
                hidden_dict = self.scatter_hiddens(hidden_dict)

            grad_norms = {}

            for names, hiddens in hidden_dict.items():
                # normalize hiddens to have unit norm
                if self.cfg.normalize_hiddens:
                    hiddens = hiddens / hiddens.norm(dim=-1, keepdim=True)

                raw = self.saes[names]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                if i == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    raw.b_dec.data = median.to(raw.dtype)

                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                    if raw.cfg.scale_encoder_fvu_global:
                        logger.info(
                            "Computing global mean and variance for FVU scaling",
                        )
                        total_variance = torch.zeros(
                            hiddens.shape[-1],
                            device=self.model.device,
                            dtype=torch.float32,
                        )
                        output_variance = torch.zeros(
                            hiddens.shape[-1],
                            device=self.model.device,
                            dtype=torch.float32,
                        )
                        test_loader = DataLoader(
                            self.test_dataset, batch_size=self.cfg.batch_size,
                        )

                        hidden_sum = torch.zeros_like(total_variance)
                        total_tokens = 0

                        for batch in test_loader:
                            hidden_dict.clear()
                            handles = [
                                mod.register_forward_hook(hook)
                                for mod in name_to_module_list.values()
                            ]
                            try:
                                with torch.no_grad():
                                    self.model(batch["input_ids"].to(device))
                            finally:
                                for handle in handles:
                                    handle.remove()

                            batch_hiddens = hidden_dict[names]
                            all_hiddens = self.maybe_all_cat(batch_hiddens)
                            hidden_sum += all_hiddens.sum(0)
                            total_tokens += all_hiddens.shape[0]

                        global_mean = hidden_sum / total_tokens

                        for batch in test_loader:
                            hidden_dict.clear()
                            handles = [
                                mod.register_forward_hook(hook)
                                for mod in name_to_module_list.values()
                            ]
                            try:
                                with torch.no_grad():
                                    self.model(batch["input_ids"].to(device))
                            finally:
                                for handle in handles:
                                    handle.remove()

                            batch_hiddens = hidden_dict[names]
                            all_hiddens = self.maybe_all_cat(batch_hiddens)

                            total_variance += ((all_hiddens - global_mean).pow(2)).sum(
                                0,
                            )

                            for chunk in all_hiddens.chunk(self.cfg.micro_acc_steps):
                                with torch.no_grad():
                                    reconstructed = raw(chunk).sae_out
                                    output_variance += (
                                        (reconstructed - chunk).pow(2)
                                    ).sum(0)

                        total_variance /= total_tokens
                        output_variance /= total_tokens

                        raw.scale_encoder_fvu(total_variance, output_variance)

                    if raw.cfg.scale_encoder_k:
                        raw.scale_encoder_k()

                all_hiddens = self.maybe_all_cat(hiddens)
                if raw.cfg.scale_encoder_fvu_batch:
                    in_var, out_var = raw.scale_encoder_fvu_batch(all_hiddens, self.cfg.micro_acc_steps)
                else:
                    in_var, out_var = raw.compute_in_out_var(all_hiddens, self.cfg.micro_acc_steps)

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[names]

                if self.cfg.tp:
                    # Each rank should receive same data
                    hiddens = self.maybe_all_cat(hiddens)

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            num_tokens_since_fired[names]
                            > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[names] += float(
                        self.maybe_all_reduce(out.fvu.detach()) / denom,
                    )
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[names] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom,
                        )

                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                    loss.div(acc_steps).backward()

                    avg_loss[names] += float(
                        self.maybe_all_reduce(loss.detach()) / denom,
                    )

                    # Update the did_fire mask
                    did_fire[names][out.latent_indices.flatten()] = True
                    self.maybe_all_reduce(
                        did_fire[names], "max",
                    )  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                grad_norms[names] = torch.nn.utils.clip_grad_norm_(
                    raw.parameters(), 1.0,
                ).item()

            # Check if we need to actually do a training step
            step, substep = divmod(i + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for names, counts in num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[names]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                info = {}

                for names in self.saes:
                    mask = (
                        num_tokens_since_fired[names] > self.cfg.dead_feature_threshold
                    )

                    names_str = "_".join(names)

                    info.update(
                        {
                            f"fvu/{names_str}": avg_fvu[names],
                            f"dead_pct/{names_str}": mask.mean(
                                dtype=torch.float32,
                            ).item(),
                            f"loss/{names_str}": avg_loss[names],
                            f"lr/{names_str}": self.optimizer.param_groups[-1]["lr"],
                            f"grad_norm/{names_str}": grad_norms[names],
                            "step": step,
                        },
                    )

                    if out_var is not None and in_var is not None:
                         info[f"in_var/{names_str}"] = in_var.cpu().item()
                         info[f"out_var/{names_str}"] = out_var.cpu().item()

                    if self.cfg.auxk_alpha > 0:
                        info[f"auxk/{names}"] = avg_auxk_loss[names]

                    # Log parameter norms
                    log_parameter_norms(self.saes[names], names_str, info)

                if (step + 1) % min(
                    self.cfg.stdout_log_frequency, self.cfg.wandb_log_frequency,
                ) == 0 and rank_zero:
                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_loss.clear()

                if (step + 1) % self.cfg.stdout_log_frequency == 0 and rank_zero:
                    logger.info(info)

                if (
                    self.cfg.log_to_wandb
                    and (step + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save(step)

        if rank_zero and self.cfg.log_to_wandb:
            wandb.finish()

        self.save(step)
        pbar.close()

    def save(self, step: int):
        """Save the SAEs to disk."""
        for hook, sae in self.saes.items():
            assert isinstance(sae, Sae)

            hook_name = "_".join(hook)

            path = self.cfg.run_name or "checkpoints"
            full_path = f"{self.cfg.root_path}/{path}/{hook_name}"

            if self.rank == 0:
                sae.save_config(full_path)

            if self.cfg.tp:
                # Save the state dict instead of pickling the entire object
                sae_state = {
                    "encoder.weight": DTensor.full_tensor(sae.encoder.weight),
                    "encoder.bias": DTensor.full_tensor(sae.encoder.bias),
                    "decoder.weight": DTensor.full_tensor(sae.W_dec),
                    "decoder.bias": DTensor.full_tensor(sae.b_dec),
                }
            else:
                sae_state = {
                    **sae.encoder.state_dict(),
                    "decoder.weight": sae.W_dec,
                    "decoder.bias": sae.b_dec,
                }

            if self.rank == 0:
                os.makedirs(full_path, exist_ok=True)
                save_file(sae_state, f"{full_path}/sae-{step}.safetensors")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()
