import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from datasets import Dataset
from einops import einsum
from torch.func import functional_call, jacrev, jvp, vjp, vmap
from torch.nn import functional as F

from src.analysis.stats import GlobalFeatureStatistics
from src.sae.utils import get_layer_list, send_to_device


@dataclass
class InterventionOutputs:
    activation_positions: Optional[torch.Tensor]
    clean_base_outputs: Optional[Any]
    intervened_later_outputs: Optional[torch.Tensor]
    input_kwargs: Any
    v_j: Optional[torch.Tensor]
    v_k: Optional[torch.Tensor]
    is_valid: Optional[torch.Tensor]


@dataclass
class CausalAttributionStrengthResult:
    proportion_explained: float
    causal_cosine: float
    error: float
    relative_error: float
    jvp: torch.Tensor
    v_j: Optional[torch.Tensor]
    v_k: Optional[torch.Tensor]
    is_valid: Optional[torch.Tensor]


def run_layers_forward_to_k(
    layers,
    j: int,
    k: int,
    input_tensor: Any,
    input_kwargs: Dict[str, Any],
    input_transform: Callable[[Any], Any] = lambda x: x,
    output_transform: Callable[[Any], Any] = lambda x: x,
):
    activations = input_transform(input_tensor)
    for layer in layers[j : k + 1]:
        params = {
            name: cast(torch.Tensor, param) for name, param in layer.named_parameters()
        }
        sig = inspect.signature(layer.forward)
        layer_kwargs = {
            name: input_kwargs[name]
            for name in sig.parameters.keys()
            if input_kwargs and name in input_kwargs
        }
        if isinstance(activations, (tuple, list)):
            out = functional_call(
                layer,
                params,
                tuple(activations),
                layer_kwargs,
            )
        else:
            out = functional_call(
                layer,
                params,
                (activations,),
                layer_kwargs,
            )
        activations = out
    return output_transform(activations)


def get_main_activation(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def compute_jacobian(model, j_activations, input_kwargs, pos, j, k):
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
    main_j_activations = get_main_activation(j_activations)
    main_j_activations.requires_grad_(True)
    _, layers = get_layer_list(model)  # type: ignore

    def forward_to_k(x):
        return run_layers_forward_to_k(
            layers,
            j,
            k,
            x,
            input_transform=lambda x: x.unsqueeze(1),
            output_transform=lambda x: x,
            input_kwargs=input_kwargs,
        )

    # Create a mask for the selected positions
    batch_size, seq_len = main_j_activations.shape[:2]
    mask = torch.zeros(
        (batch_size, seq_len), device=main_j_activations.device, dtype=torch.bool
    )
    mask[pos[:, 0], pos[:, 1]] = True

    # Select activations for specified positions
    selected_activations = main_j_activations * mask.unsqueeze(-1)

    # Compute Jacobian
    jacobian = vmap(jacrev(forward_to_k))(selected_activations)

    return jacobian.squeeze()


def compute_jvp(model, j_activations, input_kwargs, j, k, v_j, sum_over_tokens=False):
    """
    Compute batched Jacobian-vector products (JVPs) of layer k's activations w.r.t. layer j's activations for the full sequence.

    Args:
    - model: The language model (GPTNeoXModel or similar)
    - j_activations: Activations of layer j (shape: [batch_size, seq_len, hidden_size])
    - j: Index of the input layer
    - k: Index of the output layer
    - v_j: The vector to compute the JVP with (shape: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size])
    - sum_over_tokens: Whether to sum over tokens or not

    Returns:
    - Batch of JVPs
    """
    main_j_activations = get_main_activation(j_activations)
    _, layers = get_layer_list(model)  # type: ignore

    def forward_to_k(x):
        return run_layers_forward_to_k(
            layers,
            j,
            k,
            x,
            input_kwargs=input_kwargs,
        )

    # Expand v_j to [batch, seq, hidden] if needed
    if v_j.shape != main_j_activations.shape:
        v_j = v_j.expand_as(main_j_activations)

    # Compute JVP for the whole sequence
    _, jvp_out = jvp(forward_to_k, (main_j_activations,), (v_j,))[:2]

    if sum_over_tokens:
        return jvp_out[0].sum(dim=1)
    else:
        return jvp_out[0]


def perform_intervention(
    model: torch.nn.Module,
    batch: Any,
    intervention_index: int,
    readout_index: int,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    lambda_value: float = 1.0,
    num_tokens: int = 1,
    feature_idx: int = 0,
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
) -> InterventionOutputs:
    """
    Perform an intervention on a model's activations using Sparse Autoencoder (SAE) features.

    Args:
        model: The PyTorch model to intervene on.
        batch: Input tensor to the model.
        intervention_index: Index of the layer to intervene on.
        readout_index: Index of the layer to read out from.
        feature_encoder_weights: Weights of the SAE encoder.
        feature_encoder_bias: Bias of the SAE encoder.
        feature_decoder_weights: Weights of the SAE decoder.
        lambda_value: Strength of the intervention (default: 1.0).
        num_tokens: Number of tokens to intervene on (default: 1).
        feature_idx: Index of the specific feature to intervene on.
        exclude_first_k_tokens: Number of tokens to exclude from the beginning (default: 0).

    Returns:
        the results of the intervention as an InterventionOutputs object
    """
    activation_positions = None
    clean_base_outputs = None
    intervened_later_outputs = None
    v_j = None
    v_k = None
    is_valid = None
    layer_offset = intervention_index
    pre_intervention_output_kwargs = {}

    num_tokens = min(
        num_tokens, max(1, batch["input_ids"].shape[0] - exclude_first_k_tokens)
    )

    def strengthen_and_capture(module, input, kwargs, output):
        nonlocal \
            activation_positions, \
            clean_base_outputs, \
            intervened_later_outputs, \
            v_j, \
            v_k, \
            is_valid, \
            pre_intervention_output_kwargs, \
            layer_offset

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
        clean_base_outputs = output

        # Capture output kwargs for jacobian computation
        pre_intervention_output_kwargs.clear()
        pre_intervention_output_kwargs.update(kwargs)

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
        v_j = feature_decoder_segment[None, feature_idx, :]

        # Select the tokens where the feature is active
        token_mask = (feature_activation[:, :, feature_idx] > 0) & sae_top_k_mask[
            :, :, feature_idx
        ]
        activation_positions = token_mask.nonzero()
        activation_positions[:, 1] += output[0].shape[1] - token_mask.shape[1]

        new_output = output[0].clone()
        intervened_later_outputs = new_output
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
        v_k = intervention_decoder_segment[None, feature_idx, :]

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

    _, layers = get_layer_list(model)  # type: ignore
    batch = send_to_device(batch, model.device)

    # Register only one hook that does both intervention and capturing
    intervention_hook = layers[intervention_index].register_forward_hook(
        strengthen_and_capture, with_kwargs=True
    )

    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    except StopIteration:
        pass

    intervention_hook.remove()

    return InterventionOutputs(
        activation_positions,
        clean_base_outputs,
        intervened_later_outputs,
        pre_intervention_output_kwargs,  # robust static kwargs for downstream
        v_j,
        v_k,
        is_valid,
    )


def test_linear_approx(
    model: torch.nn.Module,
    dataset: Dataset,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    j: int,
    k: int,
    lam: float,
    feature_idx: int = 0,
):
    # Collect activations from GPT2
    sample = dataset[0:2]["input_ids"]

    # Perform intervention
    intervention = perform_intervention(
        model=model,
        batch={
            "input_ids": sample.cuda(),
            "attention_mask": torch.ones_like(sample, device="cuda"),
        },
        intervention_index=j,
        readout_index=k,
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        lambda_value=lam,
        num_tokens=1,
        feature_idx=feature_idx,
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
        intervention_index=j,
        readout_index=k,
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        lambda_value=0,
        num_tokens=1,
        feature_idx=feature_idx,
        exclude_first_k_tokens=0,
    )

    # Compute Jacobian
    jacobian = compute_jacobian(
        model,
        intervention.clean_base_outputs,
        intervention.activation_positions,
        intervention.input_kwargs,
        j,
        k,
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
    dataset_or_batch: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    lambda_value: float = 1.0,
    feature_idx: int = 0,
    num_tokens: int = 1,
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
    batch_size: int = 64,
):
    """
    Compute causal attribution strength for a specific latent feature index.
    Args:
        feature_idx: The index of the latent feature to analyze.
        dataset: The input data, either a dict or a Dataset.
    """

    def perform_intervention_and_compute_jvp(batch: Dict[str, torch.Tensor]):
        intervention = perform_intervention(
            model=model,
            batch=batch,
            intervention_index=j,
            readout_index=k,
            feature_encoder_weights=feature_encoder_weights,
            feature_encoder_bias=feature_encoder_bias,
            feature_decoder_weights=feature_decoder_weights,
            lambda_value=lambda_value,
            num_tokens=num_tokens,
            feature_idx=feature_idx,
            exclude_first_k_tokens=exclude_first_k_tokens,
            sae_top_k=sae_top_k,
        )

        jvp = compute_jvp(
            model,
            intervention.clean_base_outputs,
            intervention.input_kwargs,
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

        return (
            proportion_explained.item(),
            causal_cosine.item(),
            error.item(),
            relative_error.item(),
        )

    if isinstance(dataset_or_batch, Dataset):
        total_proportion_explained = 0
        total_causal_cosine = 0
        total_error = 0
        total_relative_error = 0

        for i in range(0, len(dataset_or_batch), batch_size):
            batch = dataset_or_batch[i : i + batch_size]
            intervention, jvp = perform_intervention_and_compute_jvp(batch)
            proportion_explained, causal_cosine, error, relative_error = (
                compute_metrics(intervention, jvp)
            )
            total_proportion_explained += proportion_explained
            total_causal_cosine += causal_cosine
            total_error += error
            total_relative_error += relative_error

        proportion_explained = total_proportion_explained / len(dataset_or_batch)
        causal_cosine = total_causal_cosine / len(dataset_or_batch)
        error = total_error / len(dataset_or_batch)
        relative_error = total_relative_error / len(dataset_or_batch)
    else:
        intervention, jvp = perform_intervention_and_compute_jvp(dataset_or_batch)
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
    dataset: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    marginalize_across_sequences: bool,
    feature_idx: int = 0,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    """
    results = {layer: {} for layer in range(num_layers)}
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            layer_results = {
                "causality": [],
                "causal_cosines": [],
                "self_cosine_similarity": [],
            }
            for feature_idx in range(feature_encoder_weights.shape[0]):
                idx = int(feature_idx)
                result = compute_causal_attribution_strength(
                    j=i,
                    k=j,
                    model=model,
                    dataset_or_batch=dataset,
                    feature_encoder_weights=feature_encoder_weights,
                    feature_encoder_bias=feature_encoder_bias,
                    feature_decoder_weights=feature_decoder_weights,
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
                valid_causality = result.proportion_explained
                valid_causal_cosines = result.causal_cosine
                feature_causality = valid_causality
                feature_causal_cosines = valid_causal_cosines
                layer_results["causality"].append(feature_causality)
                layer_results["causal_cosines"].append(feature_causal_cosines)
                layer_results["self_cosine_similarity"].append(
                    torch.tensor(self_cosine_sim)
                )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    if marginalize_across_sequences:
                        results[i][(i, j)][metric] = stacked_values.mean().item()
                        results[i][(i, j)][f"{metric}_std"] = (
                            stacked_values.std().item()
                        )
                    else:
                        results[i][(i, j)][metric] = np.nan
                        results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_feature_index(
    model: torch.nn.Module,
    dataset: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    feature_index: int,
    marginalize_across_sequences: bool,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    """
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
            result = compute_causal_attribution_strength(
                j=i,
                k=j,
                model=model,
                dataset_or_batch=dataset,
                feature_encoder_weights=feature_encoder_weights,
                feature_encoder_bias=feature_encoder_bias,
                feature_decoder_weights=feature_decoder_weights,
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
            valid_causality = result.proportion_explained
            valid_causal_cosines = result.causal_cosine
            feature_causality = valid_causality
            feature_causal_cosines = valid_causal_cosines
            layer_results["causality"].append(feature_causality)
            layer_results["causal_cosines"].append(feature_causal_cosines)
            layer_results["self_cosine_similarity"].append(
                torch.tensor(self_cosine_sim)
            )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    if marginalize_across_sequences:
                        results[i][(i, j)][metric] = stacked_values.mean().item()
                        results[i][(i, j)][f"{metric}_std"] = (
                            stacked_values.std().item()
                        )
                    else:
                        results[i][(i, j)][metric] = np.nan
                        results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_fixed_i(
    model: torch.nn.Module,
    dataset: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    num_layers: int,
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    fixed_i: int,
    marginalize_across_sequences: bool,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    """
    results = {layer: {} for layer in range(num_layers)}
    i = int(fixed_i)
    hidden_size = int(model.config.hidden_size)  # type: ignore
    for j in range(i + 1, num_layers):
        layer_results = {
            "causality": [],
            "causal_cosines": [],
            "self_cosine_similarity": [],
        }
        for feature_idx in range(feature_encoder_weights.shape[0]):
            idx = int(feature_idx)
            result = compute_causal_attribution_strength(
                j=i,
                k=j,
                model=model,
                dataset_or_batch=dataset,
                feature_encoder_weights=feature_encoder_weights,
                feature_encoder_bias=feature_encoder_bias,
                feature_decoder_weights=feature_decoder_weights,
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
            valid_causality = result.proportion_explained
            valid_causal_cosines = result.causal_cosine
            feature_causality = valid_causality
            feature_causal_cosines = valid_causal_cosines
            layer_results["causality"].append(feature_causality)
            layer_results["causal_cosines"].append(feature_causal_cosines)
            layer_results["self_cosine_similarity"].append(
                torch.tensor(self_cosine_sim)
            )
        results[i][(i, j)] = {}
        for metric, values in layer_results.items():
            if values:
                stacked_values = torch.cat(values)
                if marginalize_across_sequences:
                    results[i][(i, j)][metric] = stacked_values.mean().item()
                    results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
                else:
                    results[i][(i, j)][metric] = np.nan
                    results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def _eval_binned_features(
    model: torch.nn.Module,
    dataset: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    num_layers: int,
    binned_features: Sequence[Union[int, torch.Tensor]],
    lambda_value: float,
    num_tokens: int,
    exclude_first_k_tokens: int,
    marginalize_across_sequences: bool,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    """
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
                result = compute_causal_attribution_strength(
                    j=i,
                    k=j,
                    model=model,
                    dataset_or_batch=dataset,
                    feature_encoder_weights=feature_encoder_weights,
                    feature_encoder_bias=feature_encoder_bias,
                    feature_decoder_weights=feature_decoder_weights,
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
                valid_causality = result.proportion_explained
                valid_causal_cosines = result.causal_cosine
                feature_causality = valid_causality
                feature_causal_cosines = valid_causal_cosines
                layer_results["causality"].append(feature_causality)
                layer_results["causal_cosines"].append(feature_causal_cosines)
                layer_results["self_cosine_similarity"].append(
                    torch.tensor(self_cosine_sim)
                )
            results[i][(i, j)] = {}
            for metric, values in layer_results.items():
                if values:
                    stacked_values = torch.cat(values)
                    if marginalize_across_sequences:
                        results[i][(i, j)][metric] = stacked_values.mean().item()
                        results[i][(i, j)][f"{metric}_std"] = (
                            stacked_values.std().item()
                        )
                    else:
                        results[i][(i, j)][metric] = np.nan
                        results[i][(i, j)][f"{metric}_std"] = np.nan
    return results


def run_layer_pair_evaluation(
    model: torch.nn.Module,
    dataset: Dataset,
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    num_layers: int,
    binned_enc_features: Optional[Sequence[Union[int, torch.Tensor]]] = None,
    marginalization_mode: str = "all_features_above_threshold",
    feature_index: Optional[int] = None,
    fixed_i: Optional[int] = None,
    lambda_value: float = 1.0,
    num_tokens: int = 1,
    exclude_first_k_tokens: int = 0,
    marginalize_across_sequences: bool = False,
) -> Dict[int, Dict[Tuple[int, int], Dict[str, float]]]:
    """
    Evaluate layer pairs with flexible marginalization schemes.
    Args:
        dataset: Either a dict (single sequence) or a Dataset (multiple sequences)
        marginalize_across_sequences: if True, aggregate results across all sequences in dataset, else use a single random example
    Returns:
        Nested dict: {i: {(i, j): {metric: value, ...}, ...}, ...}
    """
    if not marginalize_across_sequences:
        dataset = dataset.select(random.sample(range(len(dataset)), 1))
    if marginalization_mode == "all_features_above_threshold":
        return _eval_all_features_above_threshold(
            model,
            dataset,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            marginalize_across_sequences,
        )
    elif marginalization_mode == "feature_index":
        assert feature_index is not None, "feature_index must be provided for this mode"
        return _eval_feature_index(
            model,
            dataset,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            feature_index,
            marginalize_across_sequences,
        )
    elif marginalization_mode == "fixed_i":
        assert fixed_i is not None, "fixed_i must be provided for this mode"
        return _eval_fixed_i(
            model,
            dataset,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            num_layers,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            fixed_i,
            marginalize_across_sequences,
        )
    elif marginalization_mode == "binned_features":
        assert binned_enc_features is not None, (
            "binned_features must be provided for this mode"
        )
        return _eval_binned_features(
            model,
            dataset,
            feature_encoder_weights,
            feature_encoder_bias,
            feature_decoder_weights,
            num_layers,
            binned_enc_features,
            lambda_value,
            num_tokens,
            exclude_first_k_tokens,
            marginalize_across_sequences,
        )
    else:
        raise ValueError(f"Unknown marginalization_mode: {marginalization_mode}")
