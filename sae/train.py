import os
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from omegaconf import DictConfig, OmegaConf
from simple_parsing import field
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel

from sae.config import SaeConfig

from .data import MemmapDataset, chunk_and_tokenize
from .logger import get_logger
from .trainer import SaeLayerRangeTrainer, SaeTrainer, TrainConfig
from .utils import get_open_port, set_seed

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

    train_test_split: float = 0.8
    """Fraction of the dataset to use for training."""

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

    port: int = field(default_factory=get_open_port)


def load_artifacts(
    args: RunConfig, rank: int | None = None
) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"} if rank is not None else "auto",
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

        # create train-test split
        if args.train_test_split > 0:
            dataset = dataset.train_test_split(
                test_size=args.train_test_split, seed=args.seed
            ).get(args.split)

        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc or os.cpu_count(),
            )
        else:
            logger.info("Dataset already tokenized; skipping tokenization.")

        dataset = dataset.with_format("torch")

        if (limit := args.max_examples) and args.max_examples > 0:
            dataset = dataset.select(range(limit))

    return model, dataset


def worker_main(
    rank: int,
    world_size: int,
    args: RunConfig,
):
    if args.ddp and world_size > 1:
        torch.cuda.set_device(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        dist.init_process_group("nccl", world_size=world_size, rank=rank)

        if rank == 0:
            logger.info(f"Using DDP across {dist.get_world_size()} GPUs.")
    if args.tp and rank == 0 and world_size > 1:
        logger.info(f"Using TP across {world_size} GPUs.")

    # set seeds
    set_seed(args.seed)

    # Awkward hack to prevent other ranks from duplicating data preprocessing
    if not dist.is_initialized() or args.tp or not args.ddp or rank == 0:
        model, dataset = load_artifacts(args, rank)

    if args.ddp and dist.is_initialized():
        dist.barrier()
        if rank != 0:
            model, dataset = load_artifacts(args, rank)
        dataset = dataset.shard(dist.get_world_size(), rank)

    total_tokens = len(dataset) * args.ctx_len

    trainer_cls = (
        SaeTrainer if not args.enable_cross_layer_training else SaeLayerRangeTrainer
    )

    logger.info(f"Training on '{args.dataset}' (split '{args.split}')")
    logger.info(f"Storing model weights in {model.dtype}")
    logger.info(f"Num tokens in dataset: {total_tokens:,}")
    trainer = trainer_cls(args, dataset, model, rank, world_size)
    logger.info(f"SAEs: {trainer.saes}")
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    world_size = torch.cuda.device_count()

    # Convert Hydra config to RunConfig
    parsed_config = OmegaConf.to_container(cfg, resolve=True)
    sae_config = parsed_config.pop("sae")
    args = RunConfig(sae=SaeConfig(**sae_config), **parsed_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = f"{args.run_name}_{timestamp}"

    if world_size > 1:
        logger.info(f"Spawning {world_size} processes")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, args),
        )
    else:
        worker_main(0, world_size, args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
