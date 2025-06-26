import logging
import os

import torch
import torch.distributed as dist


class DistributedAwareLogger(logging.Logger):
    """Enhanced distributed-aware logger that works with Hydra's colorlog."""

    def __init__(self, name):
        super().__init__(name)
        self.log_on_all_ranks = int(os.environ.get("LOG_ON_ALL_RANKS", 0)) == 1

    def _should_log(self):
        """Determine if this rank should log."""
        if not dist.is_initialized():
            return True

        rank = dist.get_rank()
        return rank == 0 or self.log_on_all_ranks

    def _add_rank_info(self, msg):
        """Add rank information to log message if distributed training is active."""
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            if world_size > 1:
                rank_info = f"[Rank {rank}/{world_size}] "
            else:
                rank_info = f"[Rank {rank}] "

            return f"{rank_info}{msg}"
        return msg

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        """Override _log to implement distributed-aware logging."""
        if self._should_log():
            # Add rank information to the message
            msg = self._add_rank_info(msg)
            super()._log(level, msg, args, exc_info, extra, stack_info)

    def debug_all_ranks(self, msg, *args, **kwargs):
        """Force debug logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.debug(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.debug(msg, *args, **kwargs)

    def info_all_ranks(self, msg, *args, **kwargs):
        """Force info logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.info(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.info(msg, *args, **kwargs)

    def warning_all_ranks(self, msg, *args, **kwargs):
        """Force warning logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.warning(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.warning(msg, *args, **kwargs)

    def error_all_ranks(self, msg, *args, **kwargs):
        """Force error logging on all ranks regardless of settings."""
        if dist.is_initialized():
            original_setting = self.log_on_all_ranks
            self.log_on_all_ranks = True
            self.error(msg, *args, **kwargs)
            self.log_on_all_ranks = original_setting
        else:
            self.error(msg, *args, **kwargs)


# Set the custom logger class as default
logging.setLoggerClass(DistributedAwareLogger)


def get_logger(name: str) -> logging.Logger:
    """
    Get a distributed-aware logger that works with Hydra's colorlog.

    Args:
        name: Logger name (typically __name__)

    Returns:
        DistributedAwareLogger instance
    """
    return logging.getLogger(name)


if os.environ.get("LOG_LEVEL", "DEBUG").upper() == "DEBUG":
    torch.set_printoptions(profile="full")
