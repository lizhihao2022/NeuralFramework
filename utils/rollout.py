# utils/rollout.py
from __future__ import annotations
from typing import Any, Optional, Callable
import torch


@torch.no_grad()
def autoregressive_rollout(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    u0: torch.Tensor,
    u: torch.Tensor,
    steps: int,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Generic autoregressive rollout on (B, N, C) tensors.

    Args:
        step_fn: one-step function u_next = step_fn(u_cur).
        u0: (B, N, C) initial field.
        steps: number of rollout steps.

    Returns:
        seq: (B, steps, N, C)
    """
    b, n, c = u0.shape
    cur = u0
    preds = []
    for i in range(steps):
        kwargs['y'] = u[:, i, :] if u is not None else None
        nxt = step_fn(cur, **kwargs)          # (B, N, C)
        preds.append(nxt.unsqueeze(1))
        cur = nxt
    return torch.cat(preds, dim=1)  # (B, steps, N, C)
