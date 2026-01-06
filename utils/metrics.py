# utils/metrics.py
from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable

import torch
import torch.nn.functional as F

from .loss import LossRecord


def _ensure_bnc(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (B, N, C)."""
    if x.ndim == 2:
        # (B, N) -> (B, N, 1)
        return x.unsqueeze(-1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected tensor with ndim 2 or 3, got shape={tuple(x.shape)}")


def _bnc_to_bchw(x: torch.Tensor, shape: Optional[List[int]]) -> torch.Tensor:
    """
    Convert (B, N, C) to (B, C, H, W) using spatial shape.

    Only supports 2D grids (len(shape) == 2).
    """
    if shape is None:
        raise ValueError("shape must be provided for 2D SSIM (e.g. [H, W]).")
    if len(shape) != 2:
        raise ValueError(f"SSIM currently supports 2D grids only, got shape={shape}.")

    x = _ensure_bnc(x)  # (B, N, C)
    b, n, c = x.shape
    h, w = shape
    if n != h * w:
        raise ValueError(
            f"shape {shape} implies {h*w} points, but got N={n} in (B, N, C)."
        )

    # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
    x_hw = x.view(b, h, w, c)
    return x_hw.permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


@torch.no_grad()
def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,  # kept for API compatibility
    **kwargs: Any,
) -> torch.Tensor:
    """
    Mean squared error over all elements.
    Works for any tensor shape (B, ...).
    """
    return F.mse_loss(pred, target, reduction="mean")


@torch.no_grad()
def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Root mean squared error."""
    return torch.sqrt(mse(pred, target, shape=shape) + 1e-12)


@torch.no_grad()
def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,  # not required, kept for symmetry
    data_range: Optional[float] = None,
    eps: float = 1e-12,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.

    Works for any shape as long as pred / target are aligned.
    """
    m = mse(pred, target)

    if data_range is None:
        # Use dynamic range from target
        L = (target.max() - target.min()).clamp_min(eps)
    else:
        L = torch.as_tensor(
            data_range, device=pred.device, dtype=pred.dtype
        ).clamp_min(eps)

    return 20.0 * torch.log10(L) - 10.0 * torch.log10(m + eps)


@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    data_range: Optional[float] = None,
    K1: float = 0.01,
    K2: float = 0.03,
    eps: float = 1e-12,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) with 3x3 average pooling.

    Expects batched 2D grids:
      - inputs: (B, N, C) or (B, H, W, C)
      - shape: [H, W] if inputs are (B, N, C)

    For non-2D data, this metric is not defined.
    """
    # Accept either (B, H, W, C) or (B, N, C)
    if pred.ndim == 4 and target.ndim == 4:
        # Assume BHWC
        b, h, w, c = pred.shape
        pred_chw = pred.permute(0, 3, 1, 2).contiguous()
        target_chw = target.permute(0, 3, 1, 2).contiguous()
    else:
        # Unified representation (B, N, C)
        pred_chw = _bnc_to_bchw(pred, shape)    # (B, C, H, W)
        target_chw = _bnc_to_bchw(target, shape)

    # Dynamic range
    if data_range is None:
        L = (target_chw.max() - target_chw.min()).clamp_min(eps)
    else:
        L = torch.as_tensor(
            data_range, device=pred_chw.device, dtype=pred_chw.dtype
        ).clamp_min(eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = F.avg_pool2d(pred_chw, 3, 1, 1)
    mu_y = F.avg_pool2d(target_chw, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred_chw * pred_chw, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target_chw * target_chw, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred_chw * target_chw, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + eps
    )
    return ssim_map.mean()


METRIC_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "mse": mse,
    "rmse": rmse,
    "psnr": psnr,
    "ssim": ssim,
}


class Evaluator:
    """
    Metric evaluator on unified (B, N, C) fields.

    Args:
        shape: spatial shape; used by metrics that need geometry (e.g. SSIM).
        metrics: custom metric dict; defaults to METRIC_REGISTRY.
        **metric_kwargs: extra keyword args passed to each metric.
    """

    def __init__(
        self,
        shape: Optional[List[int]] = None,
        metrics: Optional[Dict[str, Callable[..., torch.Tensor]]] = None,
        **metric_kwargs: Any,
    ) -> None:
        self.shape = shape
        self.metrics = metrics if metrics is not None else METRIC_REGISTRY
        self.kw = metric_kwargs

    def init_record(self, extra_keys: Optional[List[str]] = None) -> LossRecord:
        keys_raw = (extra_keys or []) + list(self.metrics.keys())
        seen = set()
        keys: List[str] = []
        for k in keys_raw:
            if k not in seen:
                seen.add(k)
                keys.append(k)
        return LossRecord(keys)

    @torch.no_grad()
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        record: Optional[LossRecord] = None,
        batch_size: Optional[int] = None,
        **batch: Any,
    ) -> Dict[str, float]:
        """
        Compute all registered metrics.

        Expected default input: (B, N, C).
        For rollout, you can flatten time first, e.g. y -> (B*S, N, C).
        """
        out: Dict[str, float] = {}
        for name, fn in self.metrics.items():
            val = fn(pred, target, shape=self.shape, **self.kw)
            out[name] = float(val.item())

        if record is not None:
            n = batch_size if batch_size is not None else pred.size(0)
            record.update(out, n=n)

        return out
