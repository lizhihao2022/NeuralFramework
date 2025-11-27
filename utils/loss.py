# utils/loss.py
from __future__ import annotations

from time import time
from typing import Callable, Dict, Any, Optional

import torch
import torch.distributed as dist


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {}


def register_loss(name: str) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """
    Decorator to register a loss function.

    Usage:
        @register_loss("l2")
        def l2_loss(pred, target, **batch):
            return torch.mean((pred - target) ** 2)
    """
    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = fn
        return fn

    return decorator


class CompositeLoss:
    """
    Composite loss that combines multiple named losses.

    Args:
        spec: dict mapping loss names to weights, e.g. {"l1": 1.0, "l2": 0.1}
        with_record: if True, maintain an internal LossRecord.
    """

    def __init__(self, spec: Dict[str, float], with_record: bool = False) -> None:
        self.spec = {k: float(v) for k, v in spec.items()}
        # 'total_loss' + all individual loss names
        self.loss_list = ["total_loss"] + list(self.spec.keys())
        self.record: Optional[LossRecord] = LossRecord(self.loss_list) if with_record else None

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        **batch: Any,
    ) -> torch.Tensor:
        """
        Compute the weighted sum of all registered losses.

        Args:
            pred: model prediction.
            target: ground truth.
            batch_size: number of samples in this batch (for averaging logs).
            **batch: extra info passed to each loss function (e.g. masks).
        """
        logs: Dict[str, float] = {}
        total = torch.zeros((), dtype=torch.float32, device=pred.device)

        for name, w in self.spec.items():
            if w == 0.0:
                continue
            if name not in LOSS_REGISTRY:
                raise KeyError(f"Loss '{name}' is not registered in LOSS_REGISTRY.")
            fn = LOSS_REGISTRY[name]
            val = fn(pred, target, **batch)  # scalar tensor
            if val.dim() != 0:
                # Ensure each loss returns a scalar
                val = val.mean()
            total = total + w * val
            logs[name] = float(val.detach().item())

        logs["total_loss"] = float(total.detach().item())

        # Update internal record if enabled
        if self.record is not None and batch_size is not None:
            self.record.update(logs, n=batch_size)

        return total  # used for backward

    def reset_record(self) -> None:
        """Reset the internal LossRecord, if any."""
        if self.record is not None:
            self.record = LossRecord(self.loss_list)

    def reduce_record(self, device: Optional[torch.device] = None) -> None:
        """
        All-reduce the internal LossRecord across DDP processes.
        """
        if self.record is not None:
            self.record.dist_reduce(device=device)

    def get_record_dict(self) -> Dict[str, float]:
        """
        Return current averaged losses as a plain dict.
        """
        if self.record is None:
            return {}
        return self.record.to_dict()


class LpLoss(object):
    """
    Lp loss with absolute or relative mode.

    By default __call__ returns relative Lp loss.
    """

    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True) -> None:
        super().__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Absolute Lp loss under uniform mesh assumption.
        """
        num_examples = x.size(0)
        # Assume uniform mesh along the first spatial dimension
        h = 1.0 / (x.size(1) - 1.0)

        diff = x.view(num_examples, -1) - y.view(num_examples, -1)
        all_norms = (h ** (self.d / self.p)) * torch.norm(diff, self.p, dim=1)

        if self.reduction:
            return all_norms.mean() if self.size_average else all_norms.sum()
        return all_norms

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Relative Lp loss: ||x - y||_p / ||y||_p.
        """
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
    """
    Track running averages of multiple named losses.
    """

    def __init__(self, loss_list) -> None:
        self.start_time = time()
        self.loss_list = list(loss_list)
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}

    def update(self, update_dict: Dict[str, float], n: int = 1) -> None:
        for key, value in update_dict.items():
            if key not in self.loss_dict:
                # Auto-add unseen keys to keep things flexible
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
        """
        All-reduce sums and counts for each loss across DDP processes.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")

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

    def __repr__(self) -> str:
        if not self.loss_list:
            return "LossRecord(empty)"
        first = self.loss_list[0]
        return f"{first}: {self.loss_dict[first].avg:.8f}"
