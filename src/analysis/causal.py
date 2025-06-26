from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
from einops import einsum
from torch.func import functional_call, jacrev, vjp, vmap


@dataclass
class InterventionOutputs:
    activation_positions: torch.Tensor
    clean_base_outputs: torch.Tensor
    intervened_later_outputs: torch.Tensor
    v_j: torch.Tensor
    v_k: torch.Tensor
    is_valid: torch.Tensor


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


def compute_jvp(model, j_activations, pos, j, k, v_j, sum_over_tokens=False):
    """
    Compute batched Jacobian-vector products (JVPs) of layer k's activations w.r.t. layer j's activations for select tokens.

    Args:
    - model: The language model (GPTNeoXModel or similar)
    - j_activations: Activations of layer j (shape: [batch_size, seq_len, hidden_size])
    - pos: Token positions (shape: [batch_size, num_selected_tokens])
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
        _, vjp_fn = vjp(forward_to_k, activation)
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
    batch: torch.Tensor,
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
    _, top_k_feature_index = torch.kthvalue(
        global_feature_activation_rate
        if global_acc_feature_activations is None
        else global_acc_feature_activations,
        k=feature_top_k,
        dim=-1,
    )

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
            feature_activation[:, :, top_k_feature_index]
            > 0 & sae_top_k_mask[:, :, top_k_feature_index]
        )
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
        intervention_hook = model.transformer.h[
            intervention_index
        ].register_forward_hook(
            partial(strengthen_specific_features, layer_offset=intervention_index)
        )
    else:
        intervention_hook = model.gpt_neox.layers[
            intervention_index
        ].register_forward_hook(
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
