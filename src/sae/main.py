from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from simple_parsing import field
from torch.multiprocessing.spawn import spawn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.quantization_config import BitsAndBytesConfig

from src.sae.config import SaeConfig, WandbConfig
from src.sae.data import MemmapDataset, chunk_and_tokenize
from src.sae.logger import get_logger
from src.sae.trainer import SaeLayerRangeTrainer, SaeTrainer, TrainConfig
from src.sae.utils import get_open_port, set_seed

logger = get_logger(__name__)


@dataclass
class RunConfig(TrainConfig):
    seed: int = field(default=42)
    """Random seed to use for training."""

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

    train_split: str = "train"
    """Dataset split to use for training."""

    train_test_split: float = 0.8
    """Fraction of the dataset to use for training."""

    ds_name: str | None = None
    """Dataset configuration/subset name (e.g., 'wikitext-2-raw-v1' for wikitext)."""
    
    dataset_config: str | None = None
    """Alias for ds_name. Dataset configuration/subset to load from HuggingFace."""
    
    streaming: bool = False
    """Enable streaming mode for datasets (avoid downloading entire dataset)."""
    
    cache_dir: str | None = None
    """Custom cache directory for datasets (useful if default location has insufficient space)."""

    ctx_len: int = 2048
    """Context length to use for training."""

    hf_token: str | None = None
    """Huggingface API token for downloading models."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_train_tokens: int = -1
    """Maximum number of tokens to use for training. Set to -1 for unlimited."""

    max_test_examples: int = -1
    """Maximum number of examples to use for testing."""

    data_preprocessing_num_proc: int = field(  # noqa: RUF009
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    # distributed
    ddp: bool = False

    port: int = field(default_factory=get_open_port)


def _resolve_cache_dir():
    """
    Returns the HuggingFace cache directory, using HF_HOME if set,
    otherwise defaults to ~/.cache/huggingface in the user's home directory.
    Ensures that '~' is expanded to the actual home directory.
    """
    cache_dir = os.getenv("HF_HOME")
    if cache_dir is None:
        # Always expanduser to avoid literal '~' in the path
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    else:
        cache_dir = os.path.expanduser(cache_dir)

    return os.path.abspath(cache_dir)


def load_artifacts(
    args: RunConfig,
    rank: int | None = None,
) -> tuple[PreTrainedModel, Dataset | MemmapDataset, Dataset | None]:
    # Determine the actual device index to use
    if rank is not None and torch.cuda.is_available():
        # Use rank modulo the number of available devices
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
    else:
        device_id = None
    
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": f"cuda:{device_id}"} if device_id is not None else "auto",
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        dtype=dtype,
        token=args.hf_token,
    )

    # Determine dataset config/subset name (prefer dataset_config, fallback to ds_name)
    dataset_subset = args.dataset_config or args.ds_name
    
    # Determine cache directory
    cache_directory = args.cache_dir if args.cache_dir else _resolve_cache_dir()
    
    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        # Convert token limit to example limit if needed
        max_examples = args.max_train_tokens // args.ctx_len if args.max_train_tokens > 0 else -1
        dataset = MemmapDataset(args.dataset, args.ctx_len, max_examples)
        logger.info(f"Loaded memmap dataset from {args.dataset}")
    else:
        # For Huggingface datasets
        if os.path.exists(args.dataset):
            dataset = load_from_disk(args.dataset, keep_in_memory=False)
            if isinstance(dataset, DatasetDict):
                dataset = dataset.get(args.split)
            logger.info(f"Loaded local dataset from {args.dataset}")
        else:
            # Load dataset from HuggingFace Hub
            dataset = load_dataset(
                args.dataset,
                name=dataset_subset,
                split=args.split,
                cache_dir=cache_directory,
                streaming=args.streaming,
            )
            if args.streaming:
                logger.info(f"Loaded hub dataset '{args.dataset}' in streaming mode (config: '{dataset_subset or 'default'}')")
            elif dataset_subset:
                logger.info(f"Loaded hub dataset '{args.dataset}' (config: '{dataset_subset}')")
            else:
                logger.info(f"Loaded hub dataset '{args.dataset}'")

        # Handle streaming vs regular datasets differently
        if args.streaming:
            # For streaming datasets, we can't do train_test_split
            # Users should specify the split they want directly
            test_dataset = None
            logger.info("Streaming mode: train_test_split disabled. Use 'split' parameter to select data split.")
        else:
            assert isinstance(dataset, Dataset)

            # create train-test split
            if args.train_test_split > 0:
                dataset_ = dataset.train_test_split(
                    test_size=args.train_test_split,
                    seed=args.seed,
                )
                dataset, test_dataset = dataset_.get(args.train_split), dataset_.get("test")

            assert dataset is not None and test_dataset is not None

        # Tokenization
        if not args.streaming:
            if "input_ids" not in dataset.column_names:
                tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
                dataset = chunk_and_tokenize(
                    dataset,
                    tokenizer,
                    max_seq_len=args.ctx_len,
                    num_proc=min(args.data_preprocessing_num_proc, os.cpu_count() or 1),
                )
                if test_dataset is not None:
                    test_dataset = chunk_and_tokenize(
                        test_dataset,
                        tokenizer,
                        max_seq_len=args.ctx_len,
                        num_proc=min(args.data_preprocessing_num_proc, os.cpu_count() or 1),
                    )
            else:
                logger.info("Dataset already tokenized; skipping tokenization.")

            dataset = dataset.with_format("torch")
            if test_dataset is not None:
                test_dataset = test_dataset.with_format("torch")

            # Limit dataset by tokens if specified
            if args.max_train_tokens > 0:
                # Calculate approximate number of examples needed
                limit = args.max_train_tokens // args.ctx_len
                dataset = dataset.select(range(min(limit, len(dataset))))
                logger.info(f"Limited training dataset to ~{limit} examples (~{args.max_train_tokens:,} tokens)")
            if test_dataset is not None and (limit := args.max_test_examples) and args.max_test_examples > 0:
                test_dataset = test_dataset.select(range(limit))
        else:
            # For streaming datasets, apply tokenization on-the-fly
            if "input_ids" not in list(dataset.features.keys()):
                tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
                logger.info("Applying tokenization to streaming dataset (on-the-fly)")
                dataset = chunk_and_tokenize(
                    dataset,
                    tokenizer,
                    max_seq_len=args.ctx_len,
                )
            else:
                logger.info("Dataset already tokenized; skipping tokenization.")
            
            # For streaming datasets, training loop will handle token limit

    return model, dataset, test_dataset


def worker_main(
    rank: int,
    world_size: int,
    args: RunConfig,
):
    if args.ddp and world_size > 1:
        # Set CUDA device for this process
        device_id = None
        if torch.cuda.is_available():
            device_id = rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        # Initialize process group with explicit device_id to avoid warnings
        dist.init_process_group(
            "nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{device_id}") if device_id is not None else None,
        )

        if rank == 0:
            logger.info(f"Using DDP across {dist.get_world_size()} GPUs.")
    if args.tp and rank == 0 and world_size > 1:
        logger.info(f"Using TP across {world_size} GPUs.")

    # set seeds
    set_seed(args.seed)

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not dist.is_initialized() or args.tp or not args.ddp or rank == 0:
        model, dataset, test_dataset = load_artifacts(args, rank)

    if args.ddp and dist.is_initialized():
        dist.barrier()
        if rank != 0:
            model, dataset, test_dataset = load_artifacts(args, rank)
        # Skip sharding for streaming datasets
        if not isinstance(dataset, IterableDataset):
            dataset = dataset.shard(dist.get_world_size(), rank)
            if test_dataset is not None:
                test_dataset = test_dataset.shard(dist.get_world_size(), rank)

    # Calculate total tokens (skip for streaming datasets)
    if isinstance(dataset, IterableDataset):
        logger.info(f"Training on '{args.dataset}' (split '{args.split}') in streaming mode")
        logger.info(f"Storing model weights in {model.dtype}")
        logger.info("Streaming mode: dataset size unknown")
    else:
        total_tokens = len(dataset) * args.ctx_len
        logger.info(f"Training on '{args.dataset}' (split '{args.split}')")
        logger.info(f"Storing model weights in {model.dtype}")
        logger.info(f"Num tokens in train dataset: {total_tokens:,}")

    if not args.enable_cross_layer_training:
        trainer = SaeTrainer(args, dataset, test_dataset, model, rank, world_size)  # type: ignore
    else:
        trainer = SaeLayerRangeTrainer(args, dataset, model, rank, world_size)  # type: ignore

    logger.info(f"SAEs: {trainer.saes}")
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(
    version_base=None,
    config_name="train",
    config_path=str(Path(__file__).parent.parent.parent / "config"),
)
def main(cfg: DictConfig):
    world_size = torch.cuda.device_count()

    # Convert Hydra config to RunConfig
    parsed_config = OmegaConf.to_container(cfg, resolve=True)
    sae_config = parsed_config.pop("sae")  # type: ignore
    wandb_config = parsed_config.pop("wandb")  # type: ignore
    args = RunConfig(
        sae=SaeConfig(**sae_config),
        wandb=WandbConfig(**wandb_config),
        **parsed_config
    )  # type: ignore

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        args.run_name = f"{args.run_name}_{timestamp}"
    else:
        args.run_name = timestamp

    if world_size > 1:
        logger.info(f"Spawning {world_size} processes")
        spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, args),
        )
    else:
        worker_main(0, world_size, args)


if __name__ == "__main__":
    load_dotenv(override=True)
    mp.set_start_method("spawn")
    main()
