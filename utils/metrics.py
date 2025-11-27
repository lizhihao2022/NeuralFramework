# utils/metrics.py
from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable

import torch
import torch.nn.functional as F

from .loss import LossRecord


@torch.no_grad()
def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Mean squared error over the whole batch.
    `shape` is currently unused and kept for API compatibility.
    """
    return F.mse_loss(pred, target, reduction="mean")


@torch.no_grad()
def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Root mean squared error: sqrt(MSE).
    """
    return torch.sqrt(mse(pred, target) + 1e-12)


@torch.no_grad()
def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    data_range: Optional[float] = None,
    eps: float = 1e-12,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.

    Args:
        pred, target: tensors in BHWC format.
        data_range: if provided, use this as L; otherwise infer from target.
    """
    # MSE is computed on original BHWC tensors
    m = mse(pred, target)

    # Convert to BCHW for consistency with image-style metrics (if needed later)
    pred_chw = pred.permute(0, 3, 1, 2)   # BCHW
    target_chw = target.permute(0, 3, 1, 2)

    if data_range is None:
        L = (target_chw.max() - target_chw.min()).clamp_min(eps)
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
    Structural Similarity Index (SSIM) with simple 3x3 average pooling.

    Args:
        pred, target: tensors in BHWC format.
        data_range: if None, infer L from target; otherwise use given range.
    """
    # Convert to BCHW
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)

    # Determine dynamic range L
    if data_range is None:
        L = (target.max() - target.min()).clamp_min(eps)
    else:
        L = torch.as_tensor(
            data_range, device=pred.device, dtype=pred.dtype
        ).clamp_min(eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

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
    Simple metric evaluator.

    Args:
        shape: spatial shape (kept for future use; metrics currently do not use it).
        metrics: optional custom metric dict; defaults to METRIC_REGISTRY.
        **metric_kwargs: extra keyword args passed to each metric (e.g. data_range).
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
        """
        Create a LossRecord with both extra keys (if any) and metric names.
        """
        extra_keys = extra_keys or []
        keys = extra_keys + list(self.metrics.keys())
        # Remove duplicates while preserving order
        seen = set()
        ordered_keys = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                ordered_keys.append(k)
        return LossRecord(ordered_keys)

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
        Compute all registered metrics for a batch.

        Args:
            pred, target: prediction and ground truth tensors.
            record: optional LossRecord to update.
            batch_size: number of samples in this batch (used as weight in record).
        """
        out: Dict[str, float] = {}
        for name, fn in self.metrics.items():
            val = fn(pred, target, shape=self.shape, **self.kw)
            out[name] = float(val.item())

        if record is not None:
            n = batch_size if batch_size is not None else 1
            record.update(out, n=n)

        return out
