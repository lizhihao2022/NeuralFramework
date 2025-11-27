# utils/ddp.py
import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> Tuple[bool, int]:
    """
    Initialize distributed training if launched with torchrun.

    This function is safe to call even if the process group is already
    initialized; in that case it will simply infer the local rank.

    Args:
        backend: Backend to use for distributed training (default: "nccl").

    Returns:
        distributed (bool): True if running in DDP (WORLD_SIZE > 1).
        local_rank (int): Local GPU index for this process (0 if not distributed).
    """
    # If the process group is already initialized, just infer local_rank
    if dist.is_available() and dist.is_initialized():
        local_rank_env = os.environ.get("LOCAL_RANK", "0")
        local_rank = int(local_rank_env)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, local_rank

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, etc.
        dist.init_process_group(backend=backend, init_method="env://")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return True, local_rank
    else:
        # Single-process / non-distributed mode
        return False, 0


def debug_barrier(tag: str) -> None:
    """
    A small helper for debugging synchronization issues.

    When the process group is initialized, all ranks will:
    - print "reached <tag>" with their rank
    - synchronize at a barrier
    - print "passed <tag>" after all ranks reach the barrier
    """
    if not (dist.is_available() and dist.is_initialized()):
        return

    rank = dist.get_rank()
    print(f"[Rank {rank}] reached {tag}", flush=True)
    dist.barrier()
    print(f"[Rank {rank}] passed {tag}", flush=True)
