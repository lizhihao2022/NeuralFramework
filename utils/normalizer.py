# utils/normalizer.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn


def _to_device_dtype(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.to(device=ref.device, dtype=ref.dtype)


def _broadcast_stats_to_x(stats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a stats tensor to be compatible with x (channel-last).

    The stats tensor is assumed to be computed as mean/std over training batch:
    stats shape == x_train.shape[1:].

    This helper prepends leading singleton dims until stats.ndim == x.ndim.

    Args:
        stats: Tensor of shape (..., C) where last dim is channels.
        x: Tensor of shape (..., C) where last dim is channels.

    Returns:
        A tensor with the same ndim as x, broadcastable with x.

    Raises:
        ValueError: if broadcasting is impossible or channel dim mismatches.
    """
    if x.dim() < stats.dim():
        raise ValueError(
            f"Stats has higher ndim than input: stats.ndim={stats.dim()}, x.ndim={x.dim()}."
        )
    if stats.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"Channel dim mismatch: stats.shape[-1]={stats.shape[-1]} vs x.shape[-1]={x.shape[-1]}."
        )

    out = stats
    while out.dim() < x.dim():
        out = out.unsqueeze(0)
    return out


class UnitGaussianNormalizer(nn.Module):
    """Per-dimension Gaussian normalizer (channel-last).

    Statistics are computed along the training batch dimension (dim=0).
    This implementation supports inputs such as:
      - (B, N, C)
      - (B, H, W, C)
      - (B, S, N, C)
    by broadcasting mean/std over extra leading dims.

    Note: Broadcasting assumes the non-batch trailing dimensions match those used
    to compute mean/std (up to extra leading dims like S).
    """

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def _select_stats(self, sample_idx: Optional[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select mean/std by sample_idx for compatibility. Most pipelines should pass None."""
        if sample_idx is None:
            return self.mean, self.std

        # Keep backward-compatible behavior with minimal assumptions.
        # - sample_idx can be a tensor or list/tuple of indices.
        if isinstance(sample_idx, (list, tuple)):
            return self.mean[sample_idx], self.std[sample_idx]

        return self.mean[sample_idx], self.std[sample_idx]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using stored mean/std (channel-last)."""
        mean, std = self._select_stats(sample_idx=None)
        mean = _to_device_dtype(mean, x)
        std = _to_device_dtype(std, x)

        mean = _broadcast_stats_to_x(mean, x)
        std = _broadcast_stats_to_x(std, x)

        return (x - mean) / (std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx: Optional[Any] = None) -> torch.Tensor:
        """Inverse normalization (channel-last).

        Args:
            x: Tensor to be denormalized. Examples: (B,N,C), (B,H,W,C), (B,S,N,C).
            sample_idx: Optional indices for compatibility with existing code.
                       If provided, mean/std are indexed first, then broadcast.

        Returns:
            Denormalized tensor with the same shape as x.

        Raises:
            ValueError: if stats cannot be broadcast to x.
        """
        mean, std = self._select_stats(sample_idx=sample_idx)
        mean = _to_device_dtype(mean, x)
        std = _to_device_dtype(std, x)

        mean = _broadcast_stats_to_x(mean, x)
        std = _broadcast_stats_to_x(std, x)

        return x * (std + self.eps) + mean


class GaussianNormalizer(nn.Module):
    """Global Gaussian normalizer using a single scalar mean/std."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        mean = torch.mean(x)
        std = torch.std(x)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using global mean/std."""
        mean = _to_device_dtype(self.mean, x)
        std = _to_device_dtype(self.std, x)
        return (x - mean) / (std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx: Optional[Any] = None) -> torch.Tensor:
        """Inverse normalization. sample_idx is ignored (kept for API compatibility)."""
        _ = sample_idx
        mean = _to_device_dtype(self.mean, x)
        std = _to_device_dtype(self.std, x)
        return x * (std + self.eps) + mean
