from dataclasses import dataclass, field
from typing import Dict, List

import torch
from tqdm import tqdm


@dataclass
class FeatureStats:
    causality: List[float] = field(default_factory=list)
    cosine: List[float] = field(default_factory=list)
    error: List[float] = field(default_factory=list)
    feature_activation_strength: List[float] = field(default_factory=list)


@dataclass
class GlobalFeatureStatistics:
    feature_activation_rate: torch.Tensor
    tokenwise_feature_activation_rate: torch.Tensor
    sequencewise_feature_activation_rate: torch.Tensor
    global_activation_mask: torch.Tensor
    acc_features: torch.Tensor
    total_active_features: float
    avg_active_features_per_token: float
    feature_dict: Dict[int, FeatureStats]
    n_tokens: int


def compute_feature_statistics(
    model,
    tokenized,
    feature_encoder_weights,
    feature_encoder_bias,
    sae_top_k: int = 128,
    batch_size: int = 256,
    exclude_first_k_tokens: int = 0,
    seq_len: int = 64,
):
    # (N,)
    global_feature_activation_frequencies = torch.zeros(
        feature_encoder_weights.shape[0], device=model.device
    )
    tokenwise_feature_activation_frequencies = torch.zeros(
        seq_len - exclude_first_k_tokens,
        feature_encoder_weights.shape[0],
        device=model.device,
    )
    sequencewise_feature_activation_frequencies = torch.zeros(
        len(tokenized), feature_encoder_weights.shape[0], device=model.device
    )
    global_feature_activation_mask = torch.zeros(
        len(tokenized),
        seq_len - exclude_first_k_tokens,
        feature_encoder_weights.shape[0],
        device="cpu",
    )

    n_tokens = 0

    global_acc_feature_activations = torch.zeros(
        feature_encoder_weights.shape[0], device=model.device
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized, batch_size=batch_size, shuffle=False
    )

    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hiddens = outputs.hidden_states

        stacked_hiddens = torch.cat(hiddens[1:], dim=-1)[:, exclude_first_k_tokens:, :]

        encoded_features = torch.einsum(
            "be,nse->nsb", feature_encoder_weights, stacked_hiddens
        )
        encoded_features = encoded_features + feature_encoder_bias.unsqueeze(
            0
        ).unsqueeze(0)

        k_th_strongest = (
            torch.topk(encoded_features, k=sae_top_k, dim=-1)
            .values[:, :, -1]
            .unsqueeze(-1)
        )

        batch_binary_mask = (encoded_features >= k_th_strongest).float()
        global_feature_activation_mask[i * batch_size : (i + 1) * batch_size, :, :] = (
            batch_binary_mask.cpu()
        )

        n_tokens += input_ids.shape[0] * (input_ids.shape[1] - exclude_first_k_tokens)
        global_feature_activation_frequencies += batch_binary_mask.sum(dim=(0, 1))
        sequencewise_feature_activation_frequencies[
            i * batch_size : (i + 1) * batch_size
        ] += batch_binary_mask.sum(dim=1)

        tokenwise_feature_activation_frequencies += batch_binary_mask.sum(dim=0)
        global_acc_feature_activations += (encoded_features * batch_binary_mask).sum(
            dim=(0, 1)
        )

    # Normalize to top-k rather not 1
    feature_activation_rate = global_feature_activation_frequencies / n_tokens
    tokenwise_feature_activation_rate = (
        (seq_len - exclude_first_k_tokens)
        * tokenwise_feature_activation_frequencies
        / n_tokens
    )
    normalized_acc_features = (
        global_acc_feature_activations / global_feature_activation_frequencies.sum()
    )
    sequencewise_feature_activation_rate = (
        sequencewise_feature_activation_frequencies / (seq_len - exclude_first_k_tokens)
    )

    total_active = global_feature_activation_frequencies.sum().item()
    avg_active_per_token = total_active / (
        len(tokenized) * (seq_len - exclude_first_k_tokens)
    )

    feature_dict = {i: FeatureStats() for i in range(feature_encoder_weights.shape[-1])}

    return GlobalFeatureStatistics(
        feature_activation_rate=feature_activation_rate,
        tokenwise_feature_activation_rate=tokenwise_feature_activation_rate,
        sequencewise_feature_activation_rate=sequencewise_feature_activation_rate,
        global_activation_mask=global_feature_activation_mask,
        acc_features=normalized_acc_features,
        total_active_features=total_active,
        avg_active_features_per_token=avg_active_per_token,
        feature_dict=feature_dict,
        n_tokens=n_tokens,
    )
