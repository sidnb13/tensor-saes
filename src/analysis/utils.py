from dataclasses import dataclass

import datasets
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.sae.data import chunk_and_tokenize
from src.sae.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SaeWeights:
    feature_encoder_weights: torch.Tensor
    feature_encoder_bias: torch.Tensor
    feature_decoder_weights: torch.Tensor
    feature_decoder_bias: torch.Tensor


def load_base_model(
    model_name: str, device: str = "cuda"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


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
    trust_remote_code: bool = True,
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
        trust_remote_code=trust_remote_code,
    )
    dataset = dataset.train_test_split(test_size=test_size, seed=seed).get("test")  # type: ignore
    dataset = dataset.select(range(max_ds_size))  # type: ignore

    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=seq_len)
    return tokenized
