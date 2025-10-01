import warnings
from dataclasses import dataclass
from typing import Optional

import datasets
import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,  # type: ignore
    PreTrainedTokenizerBase,  # type: ignore
)

from src.analysis.stats import GlobalFeatureStatistics
from src.sae.data import chunk_and_tokenize
from src.sae.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SaeWeights:
    feature_encoder_weights: torch.Tensor
    feature_encoder_bias: torch.Tensor
    feature_decoder_weights: torch.Tensor
    feature_decoder_bias: Optional[torch.Tensor]


def load_base_model(
    model_name: str, device: str = "cuda"
) -> tuple[AutoModelForCausalLM, PretrainedConfig, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, attn_implementation="eager"
    )
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, config, tokenizer

def deprecated(message):
  def deprecated_decorator(func):
      def deprecated_func(*args, **kwargs):
          warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                        category=DeprecationWarning,
                        stacklevel=2)
          warnings.simplefilter('default', DeprecationWarning)
          return func(*args, **kwargs)
      return deprecated_func
  return deprecated_decorator

def load_sae_from_ckpt(ckpt_path: str, device: str = "cuda") -> SaeWeights:
    sae_ckpt = load_file(ckpt_path, device=device)
    feature_encoder_weights = sae_ckpt.get("encoder.weight", sae_ckpt.get("weight"))
    feature_encoder_bias = sae_ckpt.get("encoder.bias", sae_ckpt.get("bias"))
    # legacy keys
    feature_decoder_weights = sae_ckpt["decoder.weight"]
    feature_decoder_bias = sae_ckpt["decoder.bias"]

    return SaeWeights(
        feature_encoder_weights=feature_encoder_weights,  # type: ignore
        feature_encoder_bias=feature_encoder_bias,  # type: ignore
        feature_decoder_weights=feature_decoder_weights,
        feature_decoder_bias=feature_decoder_bias,
    )


def filter_inactive_features(
    stats: GlobalFeatureStatistics,
    sae_weights: SaeWeights,
    min_activation_rate: float = 0.01,
    strategy: str = "activation_rate",
    acc_features_threshold: float = 1e-3,
) -> tuple[SaeWeights, GlobalFeatureStatistics]:
    """
    Filter out features using the specified strategy.

    Args:
        stats (GlobalFeatureStatistics): Statistics containing feature metrics.
        sae_weights (SaeWeights): SAE weights dataclass.
        min_activation_rate (float): Minimum activation rate threshold (for 'activation_rate' strategy).
        strategy (str): Filtering strategy, either 'activation_rate' or 'acc_features'.
        acc_features_threshold (float): Threshold for acc_features (for 'acc_features' strategy).

    Returns:
        SaeWeights: Filtered SaeWeights dataclass with only active features.
    """
    if strategy == "activation_rate":
        active_features = torch.where(
            stats.feature_activation_rate > min_activation_rate
        )[0]
    elif strategy == "acc_features":
        active_features = torch.where(stats.acc_features > acc_features_threshold)[0]
    else:
        raise ValueError(f"Unknown filtering strategy: {strategy}")

    filtered_feature_encoder_weights = sae_weights.feature_encoder_weights[
        active_features
    ]
    filtered_feature_encoder_bias = sae_weights.feature_encoder_bias[active_features]
    filtered_feature_decoder_weights = sae_weights.feature_decoder_weights[
        active_features
    ]
    filtered_feature_decoder_bias = (
        sae_weights.feature_decoder_bias[active_features]
        if sae_weights.feature_decoder_bias is not None
        else None
    )

    filtered_sae_weights = SaeWeights(
        feature_encoder_weights=filtered_feature_encoder_weights,
        feature_encoder_bias=filtered_feature_encoder_bias,
        feature_decoder_weights=filtered_feature_decoder_weights,
        feature_decoder_bias=filtered_feature_decoder_bias,  # type: ignore
    )

    filtered_stats = GlobalFeatureStatistics(
        feature_activation_rate=stats.feature_activation_rate[active_features.cpu()],
        global_activation_mask=stats.global_activation_mask[active_features.cpu()],
        acc_features=stats.acc_features[active_features.cpu()],
        total_active_features=stats.total_active_features,
        avg_active_features_per_token=stats.avg_active_features_per_token,
        feature_dict=stats.feature_dict,
        n_tokens=stats.n_tokens,
    )

    return filtered_sae_weights, filtered_stats


def bin_features_by_layer(feature_encoder_weights, feature_decoder_weights, num_layers):
    def calculate_layer_norms(weights, num_layers):
        """Calculate norms for each layer of the given weights."""
        layer_size = weights.shape[1] // num_layers
        layer_weights = torch.split(weights, layer_size, dim=1)
        return torch.stack([layer.norm(dim=1) for layer in layer_weights])

    """Bin features by their max norm layer for both encoder and decoder."""
    enc_norms = calculate_layer_norms(feature_encoder_weights, num_layers)
    dec_norms = calculate_layer_norms(feature_decoder_weights, num_layers)

    enc_max_norm_layers = enc_norms.argmax(dim=0).tolist()
    dec_max_norm_layers = dec_norms.argmax(dim=0).tolist()

    enc_layer_features = [[] for _ in range(num_layers)]
    dec_layer_features = [[] for _ in range(num_layers)]

    for feature_idx, (enc_layer_idx, dec_layer_idx) in enumerate(
        zip(enc_max_norm_layers, dec_max_norm_layers)
    ):
        enc_layer_features[enc_layer_idx].append(feature_idx)
        dec_layer_features[dec_layer_idx].append(feature_idx)

    enc_layer_features = [
        torch.tensor(features, device=feature_encoder_weights.device)
        for features in enc_layer_features
    ]
    dec_layer_features = [
        torch.tensor(features, device=feature_decoder_weights.device)
        for features in dec_layer_features
    ]

    return enc_layer_features, dec_layer_features


def calculate_layer_norms(weights: torch.Tensor, num_layers: int) -> torch.Tensor:
    """Calculate norms for each layer of the given weights."""
    layer_size = weights.shape[1] // num_layers
    layer_weights = torch.split(weights, layer_size, dim=1)
    return torch.stack([layer.norm(dim=1) for layer in layer_weights])


def filter_features_by_layer(
    encoder_norms: torch.Tensor, decoder_norms: torch.Tensor, layer_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter features where the specified layer has the largest norm."""
    max_norm_layers = encoder_norms.argmax(dim=0)
    layer_mask = max_norm_layers == layer_index
    return encoder_norms[:, layer_mask], decoder_norms[:, layer_mask], layer_mask


def set_layer_weights(model: AutoModelForCausalLM, k: int, value: float = 0.0):
    """
    Sets all weights in layer k and above to a specified value for the Pythia 70m model.

    Args:
    model (AutoModelForCausalLM): The Pythia 70m model
    k (int): The index of the first layer to modify
    value (float): The value to set the weights to (default is 0.0)
    """
    # Ensure k is valid
    num_layers = model.config.num_hidden_layers  # type: ignore
    if k < 0 or k >= num_layers:
        raise ValueError(f"k must be between 0 and {num_layers - 1}")

    # Set weights for transformer layers
    for i in range(k, num_layers):
        layer = model.gpt_neox.layers[i]  # type: ignore
        for param in layer.parameters():
            param.data.fill_(value)

    # Set weights for the final layer norm
    if k == 0:
        for param in model.gpt_neox.final_layer_norm.parameters():  # type: ignore
            param.data.fill_(value)

    # Set weights for the output layer (lm_head)
    model.embed_out.weight.data.fill_(value)  # type: ignore

    logger.debug(
        f"Set weights to {value} for layers {k} to {num_layers - 1}, final layer norm (if k=0), and lm_head"
    )


def load_dataset_simple(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "togethercomputer/RedPajama-Data-1T-Sample",
    split: str = "train",
    test_size: float = 0.8,
    seed: int = 42,
    max_ds_size: int = 1_000,
    seq_len: int = 64,
):
    """
    Loads a dataset from the Hugging Face Hub.

    Args:
    dataset_name (str): The name of the dataset to load.
    """
    dataset = datasets.load_dataset(
        dataset_name,
        split=split,
    )
    dataset = dataset.train_test_split(test_size=test_size, seed=seed).get("test")  # type: ignore
    dataset = dataset.select(range(max_ds_size))  # type: ignore

    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=seq_len)
    return tokenized


def create_random_sae_weights(
    num_latents: int, d_in: int, device: str = "cpu"
) -> SaeWeights:
    """
    Create random SAE weights for debug mode.
    Args:
        num_latents (int): Number of latent features.
        d_in (int): Input dimension.
        device (str): Device to create tensors on.
    Returns:
        SaeWeights: Randomly initialized SAE weights.
    """
    feature_encoder_weights = torch.randn(num_latents, d_in, device=device)
    feature_encoder_bias = torch.zeros(num_latents, device=device)
    feature_decoder_weights = torch.randn(num_latents, d_in, device=device)
    feature_decoder_bias = torch.zeros(num_latents, device=device)
    return SaeWeights(
        feature_encoder_weights=feature_encoder_weights,
        feature_encoder_bias=feature_encoder_bias,
        feature_decoder_weights=feature_decoder_weights,
        feature_decoder_bias=feature_decoder_bias,
    )
