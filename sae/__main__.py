import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count

import hydra
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from simple_parsing import field
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from sae.config import SaeConfig

from .data import MemmapDataset, chunk_and_tokenize
from .trainer import SaeLayerRangeTrainer, SaeTrainer, TrainConfig


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="gpt2",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="togethercomputer/RedPajama-Data-1T-Sample",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""
    
    ds_name: str | None = None
    """Dataset name to use when loading from huggingface."""

    ctx_len: int = 2048
    """Context length to use for training."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int = -1
    """Maximum number of examples to use for training."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    # distributed
    ddp: bool = False


def load_artifacts(
    args: RunConfig, rank: int
) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        token=args.hf_token,
    )

    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        # For Huggingface datasets
        try:
            dataset = load_dataset(
                args.dataset,
                name=args.ds_name,
                split=args.split,
                # TODO: Maybe set this to False by default? But RPJ requires it.
                trust_remote_code=True,
            )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e

        assert isinstance(dataset, Dataset)
        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc or os.cpu_count(),
            )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        dataset = dataset.with_format("torch")

        if (limit := args.max_examples) and args.max_examples > 0:
            dataset = dataset.select(range(limit))

    return model, dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    local_rank = os.environ.get("LOCAL_RANK", 0)
    ddp, tp = cfg.ddp, cfg.tp
    rank = int(local_rank)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")
    if tp and rank == 0:
        print(f"Using TP across {world_size} GPUs.")

    # Convert Hydra config to RunConfig
    parsed_config = OmegaConf.to_container(cfg, resolve=True)
    sae_config = parsed_config.pop("sae")
    args = RunConfig(sae=SaeConfig(**sae_config), **parsed_config)

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.run_name}_{timestamp}"
    else:
        timestamp = None

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if tp or not ddp or rank == 0:
        model, dataset = load_artifacts(args, rank)
    if ddp:
        dist.barrier()
        if rank != 0:
            model, dataset = load_artifacts(args, rank)
        dataset = dataset.shard(dist.get_world_size(), rank)

    total_tokens = len(dataset) * args.ctx_len

    trainer_cls = (
        SaeTrainer if not args.enable_cross_layer_training else SaeLayerRangeTrainer
    )

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")
        print(f"Num tokens in dataset: {total_tokens:,}")
        trainer = trainer_cls(args, dataset, model, rank, world_size)
        print(f"SAEs: {trainer.saes}")
        trainer.fit()


if __name__ == "__main__":
    run()
