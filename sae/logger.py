import logging
import os

import torch
import torch.distributed as dist


class DistributedAwareLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.log_on_all_ranks = (
            os.environ.get("LOG_ALL_RANKS", "False").lower() == "true"
        )

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        if extra is None:
            extra = {}
        if "rank" not in extra:
            extra["rank"] = dist.get_rank() if dist.is_initialized() else 0

        if not dist.is_initialized() or dist.get_rank() == 0 or self.log_on_all_ranks:
            super()._log(level, msg, args, exc_info, extra, stack_info)


def get_logger(name):
    logging.setLoggerClass(DistributedAwareLogger)
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get("LOG_LEVEL", "DEBUG").upper())

    return logger


if os.environ.get("LOG_LEVEL", "DEBUG").upper() == "DEBUG":
    torch.set_printoptions(profile="full")
