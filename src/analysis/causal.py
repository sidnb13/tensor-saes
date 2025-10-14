import inspect
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from datasets import Dataset
from einops import einsum
from torch.func import functional_call, jacrev, jvp
from torch.nn import functional as F
from tqdm import tqdm

from src.analysis.utils import deprecated
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
):
    activations = input_tensor
    for layer in layers[j + 1 : k + 1]:
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
    return activations


def get_main_activation(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@deprecated("Use compute_jvp instead")
def compute_jacobian(model, j_activations, input_kwargs, j, k):
    """
    Compute batched Jacobians of layer k's activations w.r.t. layer j's activations for select tokens.

    Args:
    - model: The language model (GPT2Model or similar)
    - j_activations: Activations of layer j (shape: [batch_size, seq_len, hidden_size])
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
            input_kwargs=input_kwargs,
        )

    # Compute Jacobian
    jacobian = jacrev(forward_to_k)(main_j_activations)

    # If jacobian is a tuple (e.g., (tensor, ...)), take the first element
    if isinstance(jacobian, tuple):
        jacobian = jacobian[0]

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
    feature_idx: int = 0,
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
    apply_to_all_tokens: bool = False,
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
        feature_idx: Index of the specific feature to intervene on.
        exclude_first_k_tokens: Number of tokens to exclude from the beginning (default: 0).
        sae_top_k: Number of top-k features to consider (default: 128).
        apply_to_all_tokens: If True, apply intervention to all tokens. If False, only apply where feature is active (default: False).

    Returns:
        the results of the intervention as an InterventionOutputs object
    """
    activation_positions = None
    clean_base_outputs = None
    intervened_later_outputs = None
    v_j = None
    v_k = None
    is_valid = None
    pre_intervention_output_kwargs = {}

    def strengthen_and_capture(module, input, kwargs, output):
        nonlocal \
            activation_positions, \
            clean_base_outputs, \
            intervened_later_outputs, \
            v_j, \
            v_k, \
            is_valid, \
            pre_intervention_output_kwargs

        embed_dim = output[0].shape[-1]
        feature_encoder_segment = feature_encoder_weights[
            :,
            intervention_index * embed_dim : (intervention_index + 1) * embed_dim,
        ]
        feature_decoder_segment = feature_decoder_weights[
            :,
            intervention_index * embed_dim : (intervention_index + 1) * embed_dim,
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

        if apply_to_all_tokens:
            # Apply intervention to all tokens (after exclusion)
            new_output[:, exclude_first_k_tokens:] += lambda_value * v_j.unsqueeze(
                1
            ).expand(-1, new_output.shape[1] - exclude_first_k_tokens, -1)
            # Create a mask for all tokens after exclusion
            all_tokens_mask = torch.ones(
                batch_size,
                new_output.shape[1] - exclude_first_k_tokens,
                device=new_output.device,
                dtype=torch.bool,
            )
            activation_positions = all_tokens_mask.nonzero()
            activation_positions[:, 1] += exclude_first_k_tokens

            # Set is_valid to cover all tokens after exclusion
            is_valid = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        exclude_first_k_tokens,
                        device=new_output.device,
                        dtype=torch.bool,
                    ),
                    all_tokens_mask,
                ],
                dim=1,
            )
        else:
            # Use existing logic with token_mask
            # Add intervention lambda * v_j to selected token positions after exclusion
            new_output[:, exclude_first_k_tokens:] += lambda_value * torch.einsum(
                "be,bs->bse", v_j, token_mask
            )

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

        new_outputs = [new_output] + list(output[1:])

        # Assign v_k
        intervention_decoder_segment = feature_decoder_weights[
            :,
            readout_index * embed_dim : (readout_index + 1) * embed_dim,
        ]
        v_k = intervention_decoder_segment[None, feature_idx, :]

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
    test_num_batches: int = 8,
):
    """
    Check consequent_embeddings ~= original_embeddings_at_the_higher_layer + jacobian @ v_j * lam
    """
    # select random batches
    indices = random.sample(range(len(dataset)), test_num_batches)
    sample = dataset[indices]["input_ids"]

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
        feature_idx=feature_idx,
        exclude_first_k_tokens=0,
        sae_top_k=128,
        apply_to_all_tokens=True,
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
        feature_idx=feature_idx,
        exclude_first_k_tokens=0,
        apply_to_all_tokens=True,
    )

    # Use efficient JVP computation for the linear approximation
    with torch.no_grad():
        assert clean_intervention.intervened_later_outputs is not None
        jvp = compute_jvp(
            model,
            intervention.clean_base_outputs,
            intervention.input_kwargs,
            j,
            k,
            intervention.v_j,
            sum_over_tokens=False,
        )
        jacobian_approx = clean_intervention.intervened_later_outputs.clone()

        # Only apply the linear approximation where the intervention was actually applied
        assert intervention.is_valid is not None
        masked_jvp = lam * jvp * intervention.is_valid.unsqueeze(-1)
        jacobian_approx += masked_jvp

    assert intervention.intervened_later_outputs is not None
    # Per-token error: mean squared error for each token (batch, seq)
    per_token_error = (
        (intervention.intervened_later_outputs - jacobian_approx) ** 2
    ).mean(dim=-1)
    # Overall error: mean over all tokens and batch
    overall_error = per_token_error.mean()

    print(f"Per-token error tensor:\n{per_token_error}")
    print(f"Overall error: {overall_error.item()}")

    assert overall_error < 1e-6, f"Error: {overall_error}"


@torch.no_grad()
def compute_causal_attribution_strength_batched(
    j: int,
    k: int,
    model: torch.nn.Module,
    dataset_or_batch: Union[Dict[str, torch.Tensor], Dataset],
    feature_encoder_weights: torch.Tensor,
    feature_encoder_bias: torch.Tensor,
    feature_decoder_weights: torch.Tensor,
    feature_indices: Optional[Sequence[int]] = None,
    lambda_value: float = 1.0,
    exclude_first_k_tokens: int = 0,
    apply_to_all_tokens: bool = False,
) -> Dict[int, Dict[str, float]]:
    """
    Compute causal attribution strength for multiple features efficiently by reusing clean activations.

    Args:
        j: Source layer index
        k: Target layer index
        model: The language model
        dataset_or_batch: Input data
        feature_encoder_weights: SAE encoder weights
        feature_encoder_bias: SAE encoder bias
        feature_decoder_weights: SAE decoder weights
        feature_indices: List of feature indices to evaluate. If None, evaluates all features.
        lambda_value: Intervention strength (not used in this batched version, kept for compatibility)
        exclude_first_k_tokens: Number of tokens to exclude from beginning
        apply_to_all_tokens: Whether to apply intervention to all tokens

    Returns:
        Dict mapping feature_idx -> {metric: value, ...}
    """
    hidden_size = int(model.config.hidden_size)  # type: ignore

    # Default to all features if not specified
    if feature_indices is None:
        feature_indices = list(range(feature_encoder_weights.shape[0]))

    # Get clean activations once (no intervention)
    clean_intervention = perform_intervention(
        model=model,
        batch=dataset_or_batch
        if not isinstance(dataset_or_batch, Dataset)
        else dataset_or_batch[0:1],
        intervention_index=j,
        readout_index=k,
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        lambda_value=0,  # No intervention for clean run
        feature_idx=0,  # Doesn't matter since lambda=0
        exclude_first_k_tokens=exclude_first_k_tokens,
        sae_top_k=128,
        apply_to_all_tokens=apply_to_all_tokens,
    )

    results = {}

    # Process each feature using the cached clean activations
    for idx in feature_indices:
        # Get decoder vectors for this feature
        dec_j = feature_decoder_weights[idx, j * hidden_size : (j + 1) * hidden_size]
        dec_k = feature_decoder_weights[idx, k * hidden_size : (k + 1) * hidden_size]

        # v_j is the direction at layer j
        v_j = dec_j.unsqueeze(0)  # shape: [1, hidden_size]

        # Compute JVP: how does perturbing layer j in direction v_j affect layer k?
        jvp = compute_jvp(
            model,
            clean_intervention.clean_base_outputs,
            clean_intervention.input_kwargs,
            j,
            k,
            v_j,
            sum_over_tokens=False,
        )

        # v_k is the expected direction at layer k
        v_k = dec_k.unsqueeze(0)  # shape: [1, hidden_size]
        v_k_norm_squared = torch.norm(v_k, p=2, dim=-1).pow(2)

        # Compute metrics
        proportion_explained = torch.einsum("bse,be->bs", jvp, v_k) / v_k_norm_squared

        causal_cosine = F.cosine_similarity(
            jvp,
            v_k.unsqueeze(0).unsqueeze(1),
            dim=-1,
        ).squeeze()

        self_cosine_sim = F.cosine_similarity(
            dec_j.unsqueeze(0), dec_k.unsqueeze(0)
        ).item()

        # Store results for this feature
        results[idx] = {
            "causality": proportion_explained.mean().item(),
            "causal_cosine": causal_cosine.mean().item(),
            "self_cosine_similarity": self_cosine_sim,
        }

    return results


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
    exclude_first_k_tokens: int = 0,
    sae_top_k: int = 128,
    batch_size: int = 64,
    apply_to_all_tokens: bool = False,
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
            feature_idx=feature_idx,
            exclude_first_k_tokens=exclude_first_k_tokens,
            sae_top_k=sae_top_k,
            apply_to_all_tokens=apply_to_all_tokens,
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
            proportion_explained.mean().item(),
            causal_cosine.mean().item(),
            error.mean().item(),
            relative_error.mean().item(),
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
    exclude_first_k_tokens: int,
    marginalize_across_sequences: bool,
    apply_to_all_tokens: bool,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    Batches over features for a given (i, j) layer pair to reduce redundant model forward passes.
    """
    results = {layer: {} for layer in range(num_layers)}
    num_features = feature_encoder_weights.shape[0]

    # Create cartesian product of (i, j) layer pairs only
    layer_pairs = [(i, j) for i in range(num_layers) for j in range(i + 1, num_layers)]

    pbar = tqdm(layer_pairs, desc="Layer pairs", leave=True)
    for i, j in pbar:
        pbar.set_postfix({"i": i, "j": j})

        # Initialize results storage for this layer pair
        layer_results = {
            "causality": [],
            "causal_cosines": [],
            "self_cosine_similarity": [],
        }

        # Use the batched function to compute all features for this layer pair efficiently
        feature_results = compute_causal_attribution_strength_batched(
            j=i,
            k=j,
            model=model,
            dataset_or_batch=dataset,
            feature_encoder_weights=feature_encoder_weights,
            feature_encoder_bias=feature_encoder_bias,
            feature_decoder_weights=feature_decoder_weights,
            feature_indices=None,  # Compute all features
            lambda_value=lambda_value,
            exclude_first_k_tokens=exclude_first_k_tokens,
            apply_to_all_tokens=apply_to_all_tokens,
        )

        # Collect results
        for idx in range(num_features):
            layer_results["causality"].append(feature_results[idx]["causality"])
            layer_results["causal_cosines"].append(
                feature_results[idx]["causal_cosine"]
            )
            layer_results["self_cosine_similarity"].append(
                feature_results[idx]["self_cosine_similarity"]
            )

        # Aggregate results for this layer pair
        results[i][(i, j)] = {}
        for metric, values in layer_results.items():
            if values:
                stacked_values = torch.tensor(values)
                results[i][(i, j)][metric] = stacked_values.mean().item()
                results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
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
    exclude_first_k_tokens: int,
    feature_index: int,
    marginalize_across_sequences: bool,
    apply_to_all_tokens: bool,
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
                exclude_first_k_tokens=exclude_first_k_tokens,
                apply_to_all_tokens=apply_to_all_tokens,
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
            layer_results["causality"].append(torch.tensor(feature_causality))
            layer_results["causal_cosines"].append(torch.tensor(feature_causal_cosines))
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
    exclude_first_k_tokens: int,
    fixed_i: int,
    marginalize_across_sequences: bool,
    apply_to_all_tokens: bool,
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
                exclude_first_k_tokens=exclude_first_k_tokens,
                apply_to_all_tokens=apply_to_all_tokens,
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
            layer_results["causality"].append(torch.tensor(feature_causality))
            layer_results["causal_cosines"].append(torch.tensor(feature_causal_cosines))
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
    exclude_first_k_tokens: int,
    marginalize_across_sequences: bool,
    apply_to_all_tokens: bool,
):
    """
    If marginalize_across_sequences is True, aggregate results over the full dataset. Otherwise, use only the provided single example.
    """
    results = {layer: {} for layer in range(num_layers)}
    hidden_size = int(model.config.hidden_size)  # type: ignore

    # Create list of all layer pairs to process
    layer_pairs = []
    for i in range(num_layers):
        layer_features: torch.Tensor = binned_features[i]  # type: ignore
        if len(layer_features) == 0:
            continue
        for j in range(i + 1, num_layers):
            layer_pairs.append((i, j, layer_features))

    # Add progress bar for layer pairs
    pbar = tqdm(layer_pairs, desc="Binned features layer pairs", leave=True)
    for i, j, layer_features in pbar:
        pbar.set_postfix({"i": i, "j": j, "n_features": len(layer_features)})

        # Convert layer_features to list of integers for batched processing
        feature_indices = [int(idx) for idx in layer_features]

        # Use batched causal attribution for all features at once
        batched_results = compute_causal_attribution_strength_batched(
            j=i,
            k=j,
            model=model,
            dataset_or_batch=dataset,
            feature_encoder_weights=feature_encoder_weights,
            feature_encoder_bias=feature_encoder_bias,
            feature_decoder_weights=feature_decoder_weights,
            feature_indices=feature_indices,
            lambda_value=lambda_value,
            exclude_first_k_tokens=exclude_first_k_tokens,
            apply_to_all_tokens=apply_to_all_tokens,
        )

        # Extract results and compute self cosine similarities
        layer_results = {
            "causality": [],
            "causal_cosines": [],
            "self_cosine_similarity": [],
        }

        for idx in feature_indices:
            feature_result = batched_results[idx]

            # Compute self cosine similarity between decoder vectors
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

            # Store results
            layer_results["causality"].append(torch.tensor(feature_result["causality"]))
            layer_results["causal_cosines"].append(
                torch.tensor(feature_result["causal_cosine"])
            )
            layer_results["self_cosine_similarity"].append(
                torch.tensor(self_cosine_sim)
            )

        # Aggregate results
        results[i][(i, j)] = {}
        for metric, values in layer_results.items():
            if values:
                stacked_values = torch.stack(values)
                if marginalize_across_sequences:
                    results[i][(i, j)][metric] = stacked_values.mean().item()
                    results[i][(i, j)][f"{metric}_std"] = stacked_values.std().item()
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
    exclude_first_k_tokens: int = 0,
    marginalize_across_sequences: bool = False,
    apply_to_all_tokens: bool = False,
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
            exclude_first_k_tokens,
            marginalize_across_sequences,
            apply_to_all_tokens=apply_to_all_tokens,
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
            exclude_first_k_tokens,
            feature_index,
            marginalize_across_sequences,
            apply_to_all_tokens=apply_to_all_tokens,
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
            exclude_first_k_tokens,
            fixed_i,
            marginalize_across_sequences,
            apply_to_all_tokens=apply_to_all_tokens,
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
            exclude_first_k_tokens,
            marginalize_across_sequences,
            apply_to_all_tokens=apply_to_all_tokens,
        )
    else:
        raise ValueError(f"Unknown marginalization_mode: {marginalization_mode}")
