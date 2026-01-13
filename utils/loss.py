# utils/loss.py
from __future__ import annotations

"""
Loss registry + config-driven composite losses.

Supports:
  - register new losses via @register_loss
  - build single/composite loss from YAML config
  - log each component loss alongside total

Loss functions must return a scalar tensor (or reducible to scalar by .mean()).
"""

from dataclasses import dataclass
from time import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist


# ============================================================
# Registry
# ============================================================

LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {}


def register_loss(name: str) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Decorator to register a loss function."""
    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


# ============================================================
# Built-in losses (you can add more here)
# ============================================================

@register_loss("l1")
def l1_loss(pred: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    return (pred - target).abs().mean()


@register_loss("mse")
def mse_loss(pred: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
    diff = pred - target
    return (diff * diff).mean()


@register_loss("rmse")
def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12, **_: Any) -> torch.Tensor:
    return torch.sqrt(mse_loss(pred, target).clamp_min(eps))


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with batch dim, got shape={tuple(x.shape)}")
    return x.reshape(x.size(0), -1)


@register_loss("rel_lp")
def rel_lp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    p: int = 2,
    eps: float = 1e-12,
    reduction: str = "mean",
    **_: Any,
) -> torch.Tensor:
    """
    Relative Lp: mean_b ||pred-target||_p / (||target||_p + eps)
    Works for (B,N,C), (B,S,N,C), etc.
    """
    if p <= 0:
        raise ValueError(f"p must be positive, got p={p}")
    pred_f = _flatten_batch(pred)
    tgt_f = _flatten_batch(target)
    diff_norm = torch.norm(pred_f - tgt_f, p=p, dim=1)
    tgt_norm = torch.norm(tgt_f, p=p, dim=1).clamp_min(eps)
    rel = diff_norm / tgt_norm

    if reduction == "none":
        return rel
    if reduction == "mean":
        return rel.mean()
    if reduction == "sum":
        return rel.sum()
    raise ValueError(f"Unknown reduction='{reduction}'. Expected 'mean'|'sum'|'none'.")


@register_loss("rel_l2")
def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return rel_lp_loss(pred, target, p=2, **kwargs)


# ============================================================
# LossRecord (keep your current version if you already customized it)
# ============================================================

class AverageRecord(object):
    """Keep running sum, count and average for scalar values."""
    def __init__(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class LossRecord:
    """Track running averages of multiple named losses."""
    def __init__(self, loss_list) -> None:
        self.start_time = time()
        self.loss_list = list(loss_list)
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}

    def update(self, update_dict: Dict[str, float], n: int = 1) -> None:
        for key, value in update_dict.items():
            if key not in self.loss_dict:
                self.loss_dict[key] = AverageRecord()
                if key not in self.loss_list:
                    self.loss_list.append(key)
            self.loss_dict[key].update(value, n)

    def format_metrics(self) -> str:
        parts = []
        for loss in self.loss_list:
            avg = self.loss_dict[loss].avg
            parts.append(f"{loss}: {avg:.8f}")
        elapsed = time() - self.start_time
        parts.append(f"Time: {elapsed:.2f}s")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, float]:
        return {loss: self.loss_dict[loss].avg for loss in self.loss_list}

    def dist_reduce(self, device: Optional[torch.device] = None) -> None:
        """All-reduce sums and counts for each loss across DDP processes."""
        if not (dist.is_available() and dist.is_initialized()):
            return

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

        for loss in self.loss_list:
            rec = self.loss_dict[loss]
            data = torch.tensor([rec.sum, rec.count], dtype=torch.float32, device=device)
            dist.all_reduce(data, op=dist.ReduceOp.SUM)

            global_sum, global_cnt = data.tolist()
            if global_cnt > 0:
                rec.sum = global_sum
                rec.count = global_cnt
                rec.avg = global_sum / global_cnt
            else:
                rec.sum = 0.0
                rec.count = 0.0
                rec.avg = 0.0

    def __str__(self) -> str:
        return self.format_metrics()


# ============================================================
# Composite loss + config parser
# ============================================================

@dataclass(frozen=True)
class LossSpec:
    name: str
    key: str
    weight: float
    fn: Callable[..., torch.Tensor]
    kwargs: Dict[str, Any]


LossCfgItem = Union[str, Mapping[str, Any]]


def _parse_loss_cfg(
    loss_cfg: Optional[Union[Mapping[str, Any], Sequence[LossCfgItem], str]],
    registry: Dict[str, Callable[..., torch.Tensor]],
    *,
    default: Optional[Sequence[LossCfgItem]] = None,
) -> List[LossSpec]:
    """Parse YAML-style loss config into LossSpec list."""
    if loss_cfg is None:
        loss_cfg = list(default) if default is not None else ["rel_l2"]

    global_kwargs: Dict[str, Any] = {}
    items: List[LossCfgItem]

    if isinstance(loss_cfg, str):
        items = [{"name": loss_cfg}]
    elif isinstance(loss_cfg, Mapping):
        cfg = dict(loss_cfg)
        global_kwargs.update(cfg.get("kwargs", {}) or {})

        if "spec" in cfg and cfg["spec"] is not None:
            spec_map = cfg["spec"]
            if not isinstance(spec_map, Mapping):
                raise TypeError("loss.spec must be a mapping {name: weight}")
            items = [{"name": str(k), "weight": float(v)} for k, v in spec_map.items()]
        elif "terms" in cfg and cfg["terms"] is not None:
            terms = cfg["terms"]
            if not isinstance(terms, list):
                raise TypeError("loss.terms must be a list")
            items = list(terms)
        elif "name" in cfg:
            items = [cfg]
        else:
            raise ValueError("Invalid loss config. Expected {name}, {terms}, or {spec}.")
    elif isinstance(loss_cfg, Sequence):
        items = list(loss_cfg)
    else:
        raise TypeError(f"Unsupported loss config type: {type(loss_cfg)}")

    specs: List[LossSpec] = []
    seen_keys: set[str] = set()

    for item in items:
        cfg_i = {"name": item} if isinstance(item, str) else dict(item)
        if not cfg_i.get("enabled", True):
            continue

        name = cfg_i.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid loss item: {cfg_i}")
        if name not in registry:
            raise KeyError(f"Unknown loss '{name}'. Available: {sorted(registry.keys())}")

        weight = float(cfg_i.get("weight", 1.0))
        key = cfg_i.get("key", name)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid loss output key: {cfg_i}")
        if key in seen_keys:
            raise ValueError(f"Duplicate loss output key '{key}' in loss config")
        seen_keys.add(key)

        kw = dict(global_kwargs)
        kw.update(cfg_i.get("kwargs", {}) or {}) # type: ignore

        specs.append(LossSpec(name=name, key=key, weight=weight, fn=registry[name], kwargs=kw))

    if not specs:
        raise ValueError("No losses enabled after parsing loss config.")
    return specs


class CompositeLoss:
    """
    Weighted sum of multiple losses.

    __call__(..., return_dict=True) returns (total, logs)
    logs include each term (by LossSpec.key) and 'total_loss'.
    """

    def __init__(self, specs: Sequence[LossSpec]) -> None:
        if not specs:
            raise ValueError("CompositeLoss requires at least one term")
        self.specs = list(specs)

    @property
    def keys(self) -> List[str]:
        return [s.key for s in self.specs]

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        return_dict: bool = False,
        **batch: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        logs: Dict[str, float] = {}
        total = torch.zeros((), dtype=torch.float32, device=pred.device)

        for spec in self.specs:
            if spec.weight == 0.0:
                continue
            # term kwargs override batch kwargs
            val = spec.fn(pred, target, **batch, **spec.kwargs)
            if val.dim() != 0:
                val = val.mean()
            total = total + spec.weight * val
            logs[spec.key] = float(val.detach().item())

        return (total, logs) if return_dict else total


def build_loss_fn(loss_cfg: Optional[Any]) -> CompositeLoss:
    """
    Build loss from config.

    YAML examples:
      loss: "rel_l2"

      loss:
        name: rel_lp
        kwargs: {p: 1}

      loss:
        terms:
          - {name: rel_l2, weight: 1.0}
          - {name: l1,     weight: 0.05}

      loss:
        spec: {rel_l2: 1.0, l1: 0.05}
    """
    specs = _parse_loss_cfg(loss_cfg, LOSS_REGISTRY, default=["rel_l2"])
    return CompositeLoss(specs)


# ============================================================
# Backward-compatible (legacy)
# ============================================================

class LpLoss(object):
    """Legacy relative Lp loss (kept for backward compatibility)."""

    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True) -> None:
        super().__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        diff_norms = torch.norm(x_flat - y_flat, self.p, dim=1)
        y_norms = torch.norm(y_flat, self.p, dim=1)
        y_norms = torch.where(y_norms == 0, torch.ones_like(y_norms), y_norms)
        rel = diff_norms / y_norms
        if self.reduction:
            return rel.mean() if self.size_average else rel.sum()
        return rel

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.rel(x, y)
