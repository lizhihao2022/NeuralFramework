# utils/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .loss import LossRecord


# ---------------------------
# Helpers for shape handling
# ---------------------------
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
        raise ValueError("shape must be provided for 2D metrics that need grids (e.g. SSIM).")
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape for SSIM, got shape={shape}.")
    b, n, c = x.shape
    h, w = shape
    if n != h * w:
        raise ValueError(f"Shape mismatch: N={n} but H*W={h*w} for shape={shape}.")
    x = x.view(b, h, w, c)          # (B, H, W, C)
    x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
    return x.contiguous()


def _bhwc_to_bchw(x: torch.Tensor, shape: Optional[List[int]]) -> torch.Tensor:
    """
    Convert (B, H, W, C) to (B, C, H, W), and validate against shape if provided.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (B,H,W,C) tensor, got shape={tuple(x.shape)}")
    b, h, w, c = x.shape
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"Expected 2D shape, got shape={shape}")
        if [h, w] != list(shape):
            raise ValueError(f"Grid mismatch: tensor HW={[h,w]} but shape={shape}")
    x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
    return x.contiguous()


def _as_bchw_2d(x: torch.Tensor, shape: Optional[List[int]]) -> torch.Tensor:
    """
    Accept x as:
      - (B, N, C)  with shape=[H,W]
      - (B, H, W, C)
    and return (B, C, H, W)
    """
    if x.ndim == 3:
        x = _ensure_bnc(x)
        return _bnc_to_bchw(x, shape)
    if x.ndim == 4:
        return _bhwc_to_bchw(x, shape)
    raise ValueError(f"Unsupported tensor shape for 2D grid metric: {tuple(x.shape)}")


# ---------------------------
# Metric implementations
# ---------------------------
@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    diff = pred - target
    return (diff * diff).mean()


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return torch.sqrt(mse(pred, target, **kwargs).clamp_min(0.0))


@torch.no_grad()
def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    eps: float = 1e-12,
    **_: Any,
) -> torch.Tensor:
    """
    PSNR = 20 log10(data_range) - 10 log10(MSE)
    If data_range is None, we estimate it from target (max - min).
    """
    m = mse(pred, target).clamp_min(eps)
    if data_range is None:
        dr = (target.max() - target.min()).clamp_min(eps)
    else:
        dr = torch.tensor(float(data_range), device=pred.device, dtype=pred.dtype).clamp_min(eps)
    return 20.0 * torch.log10(dr) - 10.0 * torch.log10(m)


def _gaussian_kernel_2d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).contiguous()  # (ws, ws)
    return kernel_2d


@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    shape: Optional[List[int]] = None,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
    **_: Any,
) -> torch.Tensor:
    """
    SSIM for 2D grids only.

    Accepts:
      - pred/target: (B, N, C) with shape=[H,W], OR (B, H, W, C)
    Computes channel-wise SSIM and averages over channels and batch.
    """
    x = _as_bchw_2d(pred, shape)     # (B,C,H,W)
    y = _as_bchw_2d(target, shape)   # (B,C,H,W)

    b, c, h, w = x.shape
    kernel_2d = _gaussian_kernel_2d(window_size, sigma, x.device, x.dtype)
    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)  # (C,1,ws,ws)

    # Depthwise conv
    mu_x = F.conv2d(x, kernel, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / den.clamp_min(eps)

    return ssim_map.mean()


# ---------------------------
# Registry + metadata
# ---------------------------
METRIC_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "mse": mse,
    "rmse": rmse,
    "psnr": psnr,
    "ssim": ssim,
}

# For early validation (strict mode)
_METRIC_REQUIRES_SHAPE_LEN: Dict[str, Optional[int]] = {
    "mse": None,
    "rmse": None,
    "psnr": None,
    "ssim": 2,
}


# ---------------------------
# Evaluator (configurable)
# ---------------------------
@dataclass(frozen=True)
class MetricSpec:
    name: str
    key: str
    fn: Callable[..., torch.Tensor]
    kwargs: Dict[str, Any]
    requires_shape_len: Optional[int] = None


MetricCfgItem = Union[str, Dict[str, Any]]


def _parse_metric_cfg(
    metric_cfg: Optional[Sequence[MetricCfgItem]],
    registry: Dict[str, Callable[..., torch.Tensor]],
    global_kwargs: Optional[Dict[str, Any]] = None,
) -> List[MetricSpec]:
    global_kwargs = dict(global_kwargs or {})
    if metric_cfg is None:
        metric_cfg = list(registry.keys())

    specs: List[MetricSpec] = []
    seen_keys: set[str] = set()

    for item in metric_cfg:
        if isinstance(item, str):
            cfg: Dict[str, Any] = {"name": item}
        elif isinstance(item, dict):
            cfg = dict(item)
        else:
            raise TypeError(f"Metric config item must be str or dict, got {type(item)}")

        if not cfg.get("enabled", True):
            continue

        name = cfg.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid metric config: {cfg}")

        if name not in registry:
            raise KeyError(f"Unknown metric '{name}'. Available: {sorted(registry.keys())}")

        key = cfg.get("key", name)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid metric output key in config: {cfg}")

        if key in seen_keys:
            raise ValueError(f"Duplicate metric output key '{key}' in metric config.")
        seen_keys.add(key)

        kw = dict(global_kwargs)
        kw.update(cfg.get("kwargs", {}) or {})

        specs.append(
            MetricSpec(
                name=name,
                key=key,
                fn=registry[name],
                kwargs=kw,
                requires_shape_len=_METRIC_REQUIRES_SHAPE_LEN.get(name),
            )
        )

    if not specs:
        raise ValueError("No metrics enabled after parsing metric config.")
    return specs


def _looks_like_grid_one_step(x: torch.Tensor, shape: Optional[List[int]]) -> bool:
    # (B, *shape, C)
    if shape is None:
        return False
    if x.ndim != len(shape) + 2:
        return False
    return list(x.shape[1 : 1 + len(shape)]) == list(shape)


def _looks_like_grid_rollout(x: torch.Tensor, shape: Optional[List[int]]) -> bool:
    # (B, S, *shape, C)
    if shape is None:
        return False
    if x.ndim != len(shape) + 3:
        return False
    return list(x.shape[2 : 2 + len(shape)]) == list(shape)


class Evaluator:
    """
    Flexible metric evaluator supporting:
      - one-step unified: (B, N, C)
      - rollout unified:  (B, S, N, C)
      - one-step grid:    (B, *shape, C)
      - rollout grid:     (B, S, *shape, C)

    It returns:
      - one-step: {metric_key: float}
      - rollout:  {
          "rollout_steps": S,
          f"{key}_rollout_mean": float,
          f"{key}_per_step": List[float] (optional),
        }
    """

    def __init__(
        self,
        shape: Optional[List[int]] = None,
        metrics: Optional[Dict[str, Callable[..., torch.Tensor]]] = None,
        metric_cfg: Optional[Sequence[MetricCfgItem]] = None,
        strict: bool = True,
        rollout_per_step: bool = True,
        **metric_kwargs: Any,
    ) -> None:
        self.shape = shape
        self.strict = strict
        self.rollout_per_step = rollout_per_step

        registry = metrics if metrics is not None else METRIC_REGISTRY
        self.metric_specs = _parse_metric_cfg(metric_cfg, registry, global_kwargs=metric_kwargs)
        self._validate_specs()

    def _validate_specs(self) -> None:
        if not self.strict:
            return
        for spec in self.metric_specs:
            req = spec.requires_shape_len
            if req is None:
                continue
            if self.shape is None:
                raise ValueError(
                    f"Metric '{spec.name}' requires shape len={req}, "
                    "but Evaluator.shape is None. Set data.shape or remove this metric."
                )
            if len(self.shape) != req:
                raise ValueError(
                    f"Metric '{spec.name}' requires shape len={req}, but got shape={self.shape}."
                )

    def init_record(self, extra_keys: Optional[List[str]] = None) -> LossRecord:
        keys_raw = (extra_keys or []) + [m.key for m in self.metric_specs]
        # ensure stable order without duplicates
        keys: List[str] = []
        seen: set[str] = set()
        for k in keys_raw:
            if k not in seen:
                keys.append(k)
                seen.add(k)
        return LossRecord(keys)

    @torch.no_grad()
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        record: Optional[LossRecord] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

        # Decide whether rollout
        is_rollout = False

        if pred.ndim == 5:
            # (B, S, *shape, C) rollout grid
            is_rollout = True
        elif pred.ndim == 4:
            # Ambiguous: could be (B, H, W, C) one-step grid OR (B, S, N, C) rollout unified
            # Only treat as one-step grid if it EXACTLY matches (B, *shape, C).
            if _looks_like_grid_one_step(pred, self.shape):
                is_rollout = False
            else:
                # Otherwise, treat as rollout unified (B,S,N,C)
                is_rollout = True
        else:
            # pred.ndim in {2,3} => one-step unified (B,N,C) etc.
            is_rollout = False

        if is_rollout:
            out = self._evaluate_rollout(pred, target)
            if record is not None:
                n = batch_size if batch_size is not None else pred.size(0)
                scalars: Dict[str, float] = {"rollout_steps": float(out["rollout_steps"])}
                for spec in self.metric_specs:
                    m = float(out[f"{spec.key}_rollout_mean"])
                    scalars[spec.key] = m              # rollout mean
                record.update(scalars, n=n)
            return out

        # one-step
        out: Dict[str, Any] = {}
        for spec in self.metric_specs:
            v = spec.fn(pred, target, shape=self.shape, **spec.kwargs)
            out[spec.key] = float(v.item())

        if record is not None:
            n = batch_size if batch_size is not None else pred.size(0)
            record.update({k: float(v) for k, v in out.items()}, n=n)

        return out

    def _evaluate_rollout(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        # rollout unified: (B,S,N,C) ; rollout grid: (B,S,*shape,C)
        if pred.ndim == 4:
            S = int(pred.shape[1])
            pred_steps = [pred[:, s] for s in range(S)]
            targ_steps = [target[:, s] for s in range(S)]
        elif _looks_like_grid_rollout(pred, self.shape):
            S = int(pred.shape[1])
            pred_steps = [pred[:, s] for s in range(S)]
            targ_steps = [target[:, s] for s in range(S)]
        else:
            raise ValueError(f"Unsupported rollout shape: {tuple(pred.shape)} with shape={self.shape}")

        per_step: Dict[str, List[float]] = {spec.key: [] for spec in self.metric_specs}

        for s in range(S):
            p = pred_steps[s]
            t = targ_steps[s]
            for spec in self.metric_specs:
                v = spec.fn(p, t, shape=self.shape, **spec.kwargs)
                per_step[spec.key].append(float(v.item()))

        out: Dict[str, Any] = {"rollout_steps": S}
        for spec in self.metric_specs:
            vals = per_step[spec.key]
            mean_val = float(sum(vals) / max(1, len(vals)))
            out[f"{spec.key}_rollout_mean"] = mean_val
            if self.rollout_per_step:
                out[f"{spec.key}_per_step"] = vals

        return out
