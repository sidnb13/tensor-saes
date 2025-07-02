from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from einops import einsum
from torch.func import functional_call, jacrev, vjp, vmap
from torch.nn import functional as F

from src.analysis.stats import GlobalFeatureStatistics


@dataclass
class InterventionOutputs:
    activation_positions: Optional[torch.Tensor]
    clean_base_outputs: Optional[torch.Tensor]
    intervened_later_outputs: Optional[torch.Tensor]
    v_j: Optional[torch.Tensor]
    v_k: Optional[torch.Tensor]
    is_valid: Optional[torch.Tensor]


@dataclass
class CausalAttributionStrengthResult:
    proportion_explained: torch.Tensor
    causal_cosine: torch.Tensor
    error: torch.Tensor
    relative_error: torch.Tensor
    jvp: torch.Tensor
    v_j: Optional[torch.Tensor]
    v_k: Optional[torch.Tensor]
    is_valid: Optional[torch.Tensor]


def compute_jacobian(model, j_activations, pos, j, k, sum_over_tokens: bool = True):
    """
    Compute batched Jacobians of layer k's activations w.r.t. layer j's activations for select tokens.

    Args:
    - model: The language model (GPT2Model or similar)
    - j_activations: Activations of layer j (shape: [batch_size, seq_len, hidden_size])
    - pos: Token positions (shape: [batch_size, num_selected_tokens])
    - j: Index of the input layer
    - k: Index of the output layer

    Returns:
    - Batch of Jacobians
    """
    j_activations.requires_grad_(True)

    def forward_to_k(x):
        activations = x.unsqueeze(1)
        for layer_idx in range(j, k + 1):
            layer, params = get_layer_and_params(model, layer_idx)
            activations = functional_call(layer, params, activations)[0]
        return activations

    def get_layer_and_params(model, layer_idx):
        if "gpt2" in model.__class__.__name__.lower():
            layer = model.transformer.h[layer_idx]
        else:
            layer = model.gpt_neox.layers[layer_idx]
        return layer, dict(layer.named_parameters())

    # Create a mask for the selected positions
    batch_size, seq_len = j_activations.shape[:2]
    mask = torch.zeros(
        (batch_size, seq_len), device=j_activations.device, dtype=torch.bool
    )
    mask[pos[:, 0], pos[:, 1]] = True

    # Select activations for specified positions
    selected_activations = j_activations * mask.unsqueeze(-1)

    # Sum the selected activations for each batch
    if sum_over_tokens:
        selected_activations = selected_activations.sum(dim=1, keepdim=True)

    # Compute Jacobian
    jacobian = vmap(jacrev(forward_to_k))(selected_activations)

    return jacobian.squeeze()


def compute_jvp(model, j_activations, j, k, v_j, sum_over_tokens=False):
    """
    Compute batched Jacobian-vector products (JVPs) of layer k's activations w.r.t. layer j's activations for select tokens.

    Args:
    - model: The language model (GPTNeoXModel or similar)
    - j_activations: Activations of layer j (shape: [batch_size, seq_len, hidden_size])
    - j: Index of the input layer
    - k: Index of the output layer
    - v_j: The vector to compute the JVP with (shape: [batch_size, hidden_size])
    - sum_over_tokens: Whether to sum over tokens or not

    Returns:
    - Batch of JVPs
    """

    def get_layer_and_params(model, layer_idx):
        if "gpt2" in model.__class__.__name__.lower():
            layer = model.transformer.h[layer_idx]
        else:
            layer = model.gpt_neox.layers[layer_idx]
        return layer, dict(layer.named_parameters())

    def forward_to_k(x):
        activations = x[None, None, :]
        for layer_idx in range(j, k + 1):
            layer, params = get_layer_and_params(model, layer_idx)
            activations = functional_call(layer, params, activations)[0]
        return activations.squeeze()

    # Compute VJP for a single token
    def single_token_vjp(activation, v):
        _, vjp_fn = vjp(forward_to_k, activation)  # type: ignore
        return vjp_fn(v)[0]

    # Flatten batch and sequence dimensions
    batch_size, seq_len, hidden_size = j_activations.shape
    flat_activations = j_activations.reshape(-1, hidden_size)
    # Expand v_j to match j_activations shape and then flatten
    flat_v_j = v_j.unsqueeze(1).expand(batch_size, seq_len, -1).reshape(-1, hidden_size)
    # Vmap over flattened batch and sequence dimensions
    flat_vjps = vmap(single_token_vjp)(flat_activations, flat_v_j)

    # Reshape back to original dimensions
    all_vjps = flat_vjps.reshape(batch_size, seq_len, hidden_size)

    if sum_over_tokens:
        return all_vjps.sum(dim=1)
    else:
        return all_vjps


def perform_intervention(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    global_feature_activation_rate: Optional[torch.Tensor],
    global_acc_feature_activations: Optional[torch.Tensor],
    intervention_index: int,
    readout_index: int,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    lambda_value: float = 1.0,
    num_tokens: int = 1,
    feature_top_k: int = 1,
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
) -> InterventionOutputs:
    """
    Perform an intervention on a model's activations using Sparse Autoencoder (SAE) features.

    Args:
        model: The PyTorch model to intervene on.
        batch: Input tensor to the model.
        global_feature_activation_rate: the global feature activation rate statistic
        global_acc_feature_activations: the global accumulated feature activations statistic
        intervention_index: Index of the layer to intervene on.
        readout_index: Index of the layer to read out from.
        feature_encoder_weights: Weights of the SAE encoder.
        feature_encoder_bias: Bias of the SAE encoder.
        feature_decoder_weights: Weights of the SAE decoder.
        lambda_value: Strength of the intervention (default: 1.0).
        num_tokens: Number of tokens to intervene on (default: 1).
        feature_top_k: Index of the specific feature to intervene on.
        exclude_first_k_tokens: Number of tokens to exclude from the beginning (default: 0).

    Returns:
        the results of the intervention as an InterventionOutputs object
    """

    activation_positions = None
    clean_base_outputs = None
    intervened_late_outputs = None
    # j < k in layer idx
    v_j = None
    v_k = None
    is_valid = None

    num_tokens = min(
        num_tokens, max(1, batch["input_ids"].shape[0] - exclude_first_k_tokens)
    )

    # Compute top-k index based on global activation rates
    # (or accumulated activation for each feature)
    tensor = (
        global_feature_activation_rate
        if global_acc_feature_activations is None
        else global_acc_feature_activations
    )
    assert tensor is not None, "Activation rate tensor must not be None"
    _, top_k_feature_index = torch.kthvalue(tensor, k=feature_top_k)

    def strengthen_specific_features(module, input, output, layer_offset=0):
        nonlocal \
            activation_positions, \
            clean_base_outputs, \
            intervened_late_outputs, \
            v_j, \
            v_k, \
            is_valid

        embed_dim = output[0].shape[-1]
        feature_encoder_segment = feature_encoder_weights[
            :,
            (intervention_index - layer_offset) * embed_dim : (
                intervention_index - layer_offset + 1
            )
            * embed_dim,
        ]
        feature_decoder_segment = feature_decoder_weights[
            :,
            (intervention_index - layer_offset) * embed_dim : (
                intervention_index - layer_offset + 1
            )
            * embed_dim,
        ]

        batch_size, seq_len, _ = output[0].shape
        clean_base_outputs = output[0]

        # Encode input activations after excluding first k tokens
        feature_activation = (
            einsum(
                output[0][:, exclude_first_k_tokens:],
                feature_encoder_segment.T,
                "b s e, e n -> b s n",
            )
            - feature_encoder_bias
        )
        sae_top_k_mask = torch.zeros(
            batch_size,
            feature_activation.shape[1],
            feature_encoder_segment.shape[0],
            device=output[0].device,
        ).bool()
        _, top_k_indices = torch.topk(feature_activation, k=sae_top_k, dim=-1)
        sae_top_k_mask.scatter_(2, top_k_indices, 1)

        # Get the decoder vectors for the specified feature index
        v_j = feature_decoder_segment[None, top_k_feature_index, :]

        # Select the tokens where the feature is active
        token_mask = (
            feature_activation[:, :, top_k_feature_index] > 0
        ) & sae_top_k_mask[:, :, top_k_feature_index]
        activation_positions = token_mask.nonzero()
        activation_positions[:, 1] += output[0].shape[1] - token_mask.shape[1]

        new_output = output[0].clone()
        intervened_late_outputs = new_output
        # Add intervention lambda * v_j to selected token positions after exclusion
        new_output[:, exclude_first_k_tokens:] += lambda_value * torch.einsum(
            "be,bs->bse", v_j, token_mask
        )
        new_outputs = [new_output] + list(output[1:])

        # Assign v_k
        intervention_decoder_segment = feature_decoder_weights[
            :,
            (readout_index - layer_offset) * embed_dim : (
                readout_index - layer_offset + 1
            )
            * embed_dim,
        ]
        v_k = intervention_decoder_segment[None, top_k_feature_index, :]

        # Check if the feature fires in any of the tokens after exclusion
        is_valid = token_mask.bool()

        # Concat some zeros to the left of token_mask
        extended_token_mask = torch.cat(
            [
                torch.zeros(
                    token_mask.shape[0],
                    exclude_first_k_tokens,
                    device=token_mask.device,
                    dtype=torch.bool,
                ),
                token_mask,
            ],
            dim=1,
        )

        # Update is_valid with the extended mask
        is_valid = extended_token_mask

        # Update activation_positions
        activation_positions = is_valid.nonzero()

        return tuple(new_outputs)

    if "gpt" in model.__class__.__name__.lower():
        intervention_hook = model.transformer.h[  # type: ignore
            intervention_index
        ].register_forward_hook(  # type: ignore
            partial(strengthen_specific_features, layer_offset=intervention_index)
        )
    else:
        intervention_hook = model.gpt_neox.layers[  # type: ignore
            intervention_index
        ].register_forward_hook(  # type: ignore
            partial(strengthen_specific_features, layer_offset=intervention_index)
        )

    with intervention_hook, torch.no_grad():
        model(**batch)

    return InterventionOutputs(
        activation_positions,
        clean_base_outputs,
        intervened_late_outputs,
        v_j,
        v_k,
        is_valid,
    )


def test_linear_approx(
    model: torch.nn.Module,
    tokenized: Dataset,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    j: int,
    k: int,
    lam: float,
    stats: GlobalFeatureStatistics,
):
    # Collect activations from GPT2
    sample = tokenized[0:2]["input_ids"]

    # Perform intervention
    intervention = perform_intervention(
        model=model,
        batch={
            "input_ids": sample.cuda(),
            "attention_mask": torch.ones_like(sample, device="cuda"),
        },
        global_feature_activation_rate=stats.feature_activation_rate,
        global_acc_feature_activations=stats.acc_features,
        intervention_index=j,
        readout_index=k,
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        lambda_value=lam,
        num_tokens=1,
        feature_top_k=1,
        exclude_first_k_tokens=0,
        sae_top_k=128,
    )

    # Perform clean run (no intervention)
    clean_intervention = perform_intervention(
        model=model,
        batch={
            "input_ids": sample.cuda(),
            "attention_mask": torch.ones_like(sample, device="cuda"),
        },
        global_feature_activation_rate=stats.feature_activation_rate,
        global_acc_feature_activations=stats.acc_features,
        intervention_index=j,
        readout_index=k,
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        lambda_value=0,
        num_tokens=1,
        feature_top_k=1,
        exclude_first_k_tokens=0,
    )

    # Compute Jacobian
    jacobian = compute_jacobian(
        model, intervention.clean_base_outputs, intervention.activation_positions, j, k
    )

    # Check consequent_embeddings ~= original_embeddings_at_the_higher_layer + jacobian @ v_j * lam
    with torch.no_grad():
        assert clean_intervention.intervened_later_outputs is not None
        jacobian_approx = clean_intervention.intervened_later_outputs.clone()

        # Create a mask for the selected token positions
        batch_size, seq_len, _ = jacobian_approx.shape
        token_mask = torch.zeros(
            batch_size, seq_len, device=jacobian_approx.device, dtype=torch.bool
        )

        assert intervention.activation_positions is not None
        assert intervention.activation_positions.shape[1] == 2
        assert intervention.activation_positions.shape[0] == batch_size
        token_mask[
            intervention.activation_positions[:, 0],
            intervention.activation_positions[:, 1],
        ] = True

        # Apply lam * JVP only at the correct activation positions
        assert intervention.v_j is not None
        assert intervention.v_j.shape[1] == 1
        jvp = lam * torch.einsum("j,bij->bi", intervention.v_j.squeeze(), jacobian)
        jacobian_approx += jvp.unsqueeze(1) * token_mask.unsqueeze(-1)

    assert intervention.intervened_later_outputs is not None
    error = torch.mean((intervention.intervened_later_outputs - jacobian_approx) ** 2)

    assert error < 1e-6, f"Error: {error}"


@torch.no_grad()
def compute_causal_attribution_strength(
    j: int,
    k: int,
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_feature_activation_rate: Optional[torch.Tensor],
    global_acc_feature_activations: Optional[torch.Tensor],
    lambda_value: float = 1.0,
    feature_idx: int = 0,
    num_tokens: int = 1,
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
):
    """
    Compute causal attribution strength for a specific latent feature index.
    Args:
        feature_idx: The index of the latent feature to analyze.
    """

    def perform_intervention_and_compute_jvp():
        intervention = perform_intervention(
            model=model,
            batch=inputs,
            global_feature_activation_rate=global_feature_activation_rate,
            global_acc_feature_activations=global_acc_feature_activations,
            intervention_index=j,
            readout_index=k,
            feature_encoder_weights=feature_encoder_weights,
            feature_encoder_bias=feature_encoder_bias,
            feature_decoder_weights=feature_decoder_weights,
            lambda_value=lambda_value,
            num_tokens=num_tokens,
            feature_top_k=feature_idx + 1,  # keep this for now, but see below
            exclude_first_k_tokens=exclude_first_k_tokens,
            sae_top_k=sae_top_k,
        )

        jvp = compute_jvp(
            model,
            intervention.clean_base_outputs,
            j,
            k,
            intervention.v_j,
            sum_over_tokens=False,
        )

        return intervention, jvp

    def compute_metrics(intervention, jvp):
        v_k_norm_squared = torch.norm(intervention.v_k, p=2, dim=-1).pow(2)

        proportion_explained = (
            torch.einsum(
                "bse,be->bs",
                jvp,
                intervention.v_k.squeeze(1),
            )
            / v_k_norm_squared
        )

        causal_cosine = F.cosine_similarity(
            jvp,  # shape: [batch, seq, embed]
            intervention.v_k.unsqueeze(0).unsqueeze(
                1
            ),  # shape: [1, 1, embed] -> [batch, seq, embed]
            dim=-1,  # compute similarity along the embed dimension
        ).squeeze()  # resulting shape: [batch, seq]

        error = torch.mean(
            (jvp - intervention.v_k.unsqueeze(1).expand_as(jvp)) ** 2, dim=(-1, -2)
        )
        relative_error = error / v_k_norm_squared.squeeze()

        return proportion_explained, causal_cosine, error, relative_error

    intervention, jvp = perform_intervention_and_compute_jvp()
    proportion_explained, causal_cosine, error, relative_error = compute_metrics(
        intervention, jvp
    )

    return CausalAttributionStrengthResult(
        proportion_explained=proportion_explained,
        causal_cosine=causal_cosine,
        error=error,
        relative_error=relative_error,
        jvp=jvp,
        v_j=intervention.v_j,
        v_k=intervention.v_k,
        is_valid=intervention.is_valid,
    )


def _eval_all_features_above_threshold(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_statistic: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    use_accumulated: bool,
    activation_threshold: float,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    results = {layer: {} for layer in range(num_layers)}
    feature_mask = global_statistic > activation_threshold
    feature_indices = torch.arange(global_statistic.shape[0])[feature_mask]
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            layer_results = {
                "causality": [],
                "causal_cosines": [],
                "self_cosine_similarity": [],
            }
            for feature_idx in feature_indices:
                idx = int(feature_idx)
                if use_accumulated:
                    global_feature_activation_rate = None
                    global_acc_feature_activations = global_statistic
                else:
                    global_feature_activation_rate = global_statistic
                    global_acc_feature_activations = None
                result = compute_causal_attribution_strength(
                    j=i,
                    k=j,
                    model=model,
                    inputs={"input_ids": input_ids.cuda()},
                    feature_encoder_weights=feature_encoder_weights,
                    feature_encoder_bias=feature_encoder_bias,
                    feature_decoder_weights=feature_decoder_weights,
                    global_feature_activation_rate=global_feature_activation_rate,
                    global_acc_feature_activations=global_acc_feature_activations,
                    lambda_value=lambda_value,
                    feature_idx=idx,
                    num_tokens=num_tokens,
                    exclude_first_k_tokens=exclude_first_k_tokens,
                )
                dec_i = feature_decoder_weights[
                    idx,
                    i * hidden_size : (i + 1) * hidden_size,
                ]
                dec_j = feature_decoder_weights[
                    idx,
                    j * hidden_size : (j + 1) * hidden_size,
                ]
                self_cosine_sim = torch.nn.functional.cosine_similarity(
                    dec_i.unsqueeze(0), dec_j.unsqueeze(0)
                ).item()
                if result.is_valid is not None and result.is_valid.any():
                    valid_causality = result.proportion_explained[result.is_valid]
                    valid_causal_cosines = result.causal_cosine[result.is_valid]
                    layer_results["causality"].append(valid_causality)
                    layer_results["causal_cosines"].append(valid_causal_cosines)
                    layer_results["self_cosine_similarity"].append(
                        torch.full((valid_causality.numel(),), self_cosine_sim)
                    )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    results[i][(i, j)][metric] = stacked_values.mean().item()
                    results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
                else:
                    results[i][(i, j)][metric] = np.nan
                    results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_feature_index(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_statistic: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    use_accumulated: bool,
    feature_index: int,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    results = {layer: {} for layer in range(num_layers)}
    idx = int(feature_index)
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            layer_results = {
                "causality": [],
                "causal_cosines": [],
                "self_cosine_similarity": [],
            }
            if use_accumulated:
                global_feature_activation_rate = None
                global_acc_feature_activations = global_statistic
            else:
                global_feature_activation_rate = global_statistic
                global_acc_feature_activations = None
            result = compute_causal_attribution_strength(
                j=i,
                k=j,
                model=model,
                inputs={"input_ids": input_ids.cuda()},
                feature_encoder_weights=feature_encoder_weights,
                feature_encoder_bias=feature_encoder_bias,
                feature_decoder_weights=feature_decoder_weights,
                global_feature_activation_rate=global_feature_activation_rate,
                global_acc_feature_activations=global_acc_feature_activations,
                lambda_value=lambda_value,
                feature_idx=idx,
                num_tokens=num_tokens,
                exclude_first_k_tokens=exclude_first_k_tokens,
            )
            dec_i = feature_decoder_weights[
                idx,
                i * hidden_size : (i + 1) * hidden_size,
            ]
            dec_j = feature_decoder_weights[
                idx,
                j * hidden_size : (j + 1) * hidden_size,
            ]
            self_cosine_sim = torch.nn.functional.cosine_similarity(
                dec_i.unsqueeze(0), dec_j.unsqueeze(0)
            ).item()
            if result.is_valid is not None and result.is_valid.any():
                valid_causality = result.proportion_explained[result.is_valid]
                valid_causal_cosines = result.causal_cosine[result.is_valid]
                layer_results["causality"].append(valid_causality)
                layer_results["causal_cosines"].append(valid_causal_cosines)
                layer_results["self_cosine_similarity"].append(
                    torch.full((valid_causality.numel(),), self_cosine_sim)
                )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    results[i][(i, j)][metric] = stacked_values.mean().item()
                    results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
                else:
                    results[i][(i, j)][metric] = np.nan
                    results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_fixed_i(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_statistic: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    use_accumulated: bool,
    activation_threshold: float,
    fixed_i: int,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    results = {layer: {} for layer in range(num_layers)}
    i = int(fixed_i)
    feature_mask = global_statistic > activation_threshold
    feature_indices = torch.arange(global_statistic.shape[0])[feature_mask]
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for j in range(i + 1, num_layers):
        layer_results = {
            "causality": [],
            "causal_cosines": [],
            "self_cosine_similarity": [],
        }
        for feature_idx in feature_indices:
            idx = int(feature_idx)
            if use_accumulated:
                global_feature_activation_rate = None
                global_acc_feature_activations = global_statistic
            else:
                global_feature_activation_rate = global_statistic
                global_acc_feature_activations = None
            result = compute_causal_attribution_strength(
                j=i,
                k=j,
                model=model,
                inputs={"input_ids": input_ids.cuda()},
                feature_encoder_weights=feature_encoder_weights,
                feature_encoder_bias=feature_encoder_bias,
                feature_decoder_weights=feature_decoder_weights,
                global_feature_activation_rate=global_feature_activation_rate,
                global_acc_feature_activations=global_acc_feature_activations,
                lambda_value=lambda_value,
                feature_idx=idx,
                num_tokens=num_tokens,
                exclude_first_k_tokens=exclude_first_k_tokens,
            )
            dec_i = feature_decoder_weights[
                idx,
                i * hidden_size : (i + 1) * hidden_size,
            ]
            dec_j = feature_decoder_weights[
                idx,
                j * hidden_size : (j + 1) * hidden_size,
            ]
            self_cosine_sim = torch.nn.functional.cosine_similarity(
                dec_i.unsqueeze(0), dec_j.unsqueeze(0)
            ).item()
            if result.is_valid is not None and result.is_valid.any():
                valid_causality = result.proportion_explained[result.is_valid]
                valid_causal_cosines = result.causal_cosine[result.is_valid]
                layer_results["causality"].append(valid_causality)
                layer_results["causal_cosines"].append(valid_causal_cosines)
                layer_results["self_cosine_similarity"].append(
                    torch.full((valid_causality.numel(),), self_cosine_sim)
                )
        results[i][(i, j)] = {}
        for metric, values in layer_results.items():
            if values:
                stacked_values = torch.cat(values)
                results[i][(i, j)][metric] = stacked_values.mean().item()
                results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
            else:
                results[i][(i, j)][metric] = np.nan
                results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_binned_features(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_statistic: torch.Tensor,
    num_layers: int,
    binned_features: Sequence[Union[int, torch.Tensor]],
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    use_accumulated: bool,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    results = {layer: {} for layer in range(num_layers)}
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for i in range(num_layers):
        layer_features: torch.Tensor = binned_features[i]  # type: ignore
        if len(layer_features) == 0:
            continue
        for j in range(i + 1, num_layers):
            layer_results = {
                "causality": [],
                "causal_cosines": [],
                "self_cosine_similarity": [],
            }
            for feature_idx in layer_features:
                idx = int(feature_idx)
                if use_accumulated:
                    global_feature_activation_rate = None
                    global_acc_feature_activations = global_statistic[layer_features]
                else:
                    global_feature_activation_rate = global_statistic[layer_features]
                    global_acc_feature_activations = None
                result = compute_causal_attribution_strength(
                    j=i,
                    k=j,
                    model=model,
                    inputs={"input_ids": input_ids.cuda()},
                    feature_encoder_weights=feature_encoder_weights,
                    feature_encoder_bias=feature_encoder_bias,
                    feature_decoder_weights=feature_decoder_weights,
                    global_feature_activation_rate=global_feature_activation_rate,
                    global_acc_feature_activations=global_acc_feature_activations,
                    lambda_value=lambda_value,
                    feature_idx=idx,
                    num_tokens=num_tokens,
                    exclude_first_k_tokens=exclude_first_k_tokens,
                )
                dec_i = feature_decoder_weights[
                    idx,
                    i * hidden_size : (i + 1) * hidden_size,
                ]
                dec_j = feature_decoder_weights[
                    idx,
                    j * hidden_size : (j + 1) * hidden_size,
                ]
                self_cosine_sim = torch.nn.functional.cosine_similarity(
                    dec_i.unsqueeze(0), dec_j.unsqueeze(0)
                ).item()
                if result.is_valid is not None and result.is_valid.any():
                    valid_causality = result.proportion_explained[result.is_valid]
                    valid_causal_cosines = result.causal_cosine[result.is_valid]
                    layer_results["causality"].append(valid_causality)
                    layer_results["causal_cosines"].append(valid_causal_cosines)
                    layer_results["self_cosine_similarity"].append(
                        torch.full((valid_causality.numel(),), self_cosine_sim)
                    )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    results[i][(i, j)][metric] = stacked_values.mean().item()
                    results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
                else:
                    results[i][(i, j)][metric] = np.nan
                    results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def run_layer_pair_evaluation(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    global_statistic: torch.Tensor,
    num_layers: int,
    binned_enc_features: Optional[Sequence[Union[int, torch.Tensor]]] = None,
    marginalization_mode: str = "all_features_above_threshold",  # or "feature_index", "fixed_i", "binned_features"
    activation_threshold: float = 0.0,
    feature_index: Optional[int] = None,
    fixed_i: Optional[int] = None,
    lambda_value: float = 1.0,
    num_tokens: int = 1,
    exclude_first_k_tokens: int = 0,
    use_accumulated: bool = False,
    marginalize_across_sequences: bool = False,
    tokenized_batch: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    """
    Evaluate layer pairs with flexible marginalization schemes, optionally across a batch of sequences.
    Args:
        input_ids: torch.Tensor of token ids (batch, seq_len)
        marginalization_mode: one of ["all_features_above_threshold", "feature_index", "fixed_i", "binned_features"]
        activation_threshold: used for all_features_above_threshold and fixed_i
        feature_index: used for feature_index mode
        fixed_i: used for fixed_i mode
        marginalize_across_sequences: if True, aggregate results across tokenized_batch
        tokenized_batch: list of dicts, each with 'input_ids' (and optionally 'attention_mask')
    Returns:
        Nested dict: {i: {(i, j): {metric: value, ...}, ...}, ...}
    """
    if marginalize_across_sequences:
        assert tokenized_batch is not None and len(tokenized_batch) > 0, "tokenized_batch must be provided if marginalizing across sequences"
        # Collect results for each sequence
        all_results = []
        for seq in tokenized_batch:
            seq_input_ids = seq["input_ids"]
            # Call recursively with marginalize_across_sequences=False
            result = run_layer_pair_evaluation(
                model,
                seq_input_ids,
                feature_encoder_weights,
                feature_encoder_bias,
                feature_decoder_weights,
                global_statistic,
                num_layers,
                binned_enc_features=binned_enc_features,
                marginalization_mode=marginalization_mode,
                activation_threshold=activation_threshold,
                feature_index=feature_index,
                fixed_i=fixed_i,
                lambda_value=lambda_value,
                num_tokens=num_tokens,
                exclude_first_k_tokens=exclude_first_k_tokens,
                use_accumulated=use_accumulated,
                marginalize_across_sequences=False,
                tokenized_batch=None,
            )
            all_results.append(result)
        # Aggregate results: for each i, (i,j), metric, collect values and compute mean/std
        # Assume all dicts have the same structure
        agg_results = {}
        for i in all_results[0]:
            agg_results[i] = {}
            for pair in all_results[0][i]:
                agg_results[i][pair] = {}
                # Find all metrics
                metrics = all_results[0][i][pair].keys()
                for metric in metrics:
                    vals = [res[i][pair][metric] for res in all_results]
                    # Only aggregate if not nan
                    vals = [v for v in vals if v == v]  # filter out nan
                    if len(vals) == 0:
                        agg_results[i][pair][metric] = float('nan')
                    else:
                        agg_results[i][pair][metric] = float(torch.tensor(vals).mean())
                        agg_results[i][pair][f"{metric}_std"] = float(torch.tensor(vals).std())
        return agg_results
    # Original logic for a single sequence
    if marginalization_mode == "all_features_above_threshold":
        return _eval_all_features_above_threshold(
            model,
            input_ids,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            global_statistic,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            use_accumulated,
            activation_threshold,
        )
    elif marginalization_mode == "feature_index":
        assert feature_index is not None, "feature_index must be provided for this mode"
        return _eval_feature_index(
            model,
            input_ids,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            global_statistic,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            use_accumulated,
            feature_index,
        )
    elif marginalization_mode == "fixed_i":
        assert fixed_i is not None, "fixed_i must be provided for this mode"
        return _eval_fixed_i(
            model,
            input_ids,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            global_statistic,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            use_accumulated,
            activation_threshold,
            fixed_i,
        )
    elif marginalization_mode == "binned_features":
        assert binned_enc_features is not None, (
            "binned_features must be provided for this mode"
        )
        return _eval_binned_features(
            model,
            input_ids,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            global_statistic,
            num_layers,
            binned_enc_features,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            use_accumulated,
        )
    else:
        raise ValueError(f"Unknown marginalization_mode: {marginalization_mode}")
