# forecastors/base.py
from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Sequence, Union

import yaml
import numpy as np
import torch
import torch.utils.data as data

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from utils import Evaluator, set_seed, get_vis_fn
from .data_bundle import DataBundle


def _prod(shape: Sequence[int]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return int(p)


def _decode_field(
    norm: Optional[Any],
    u: torch.Tensor,
    shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """Decode a field tensor using a normalizer fit on (B, P, C).

    Supports:
      - one-step unified: (B, P, C)
      - rollout unified:  (B, S, P, C)
      - one-step grid:    (B, *shape, C)
      - rollout grid:     (B, S, *shape, C)

    If `shape` is None, falls back to norm.decode(u).
    """
    if norm is None or not hasattr(norm, "decode"):
        return u
    if shape is None:
        return norm.decode(u)

    P = _prod(shape)

    # one-step unified
    if u.ndim == 3 and u.shape[1] == P:
        return norm.decode(u)

    # rollout unified
    if u.ndim == 4 and u.shape[2] == P:
        B, S, _, C = u.shape
        flat = u.reshape(B * S, P, C)
        dec = norm.decode(flat)
        return dec.reshape(B, S, P, C)

    spatial_ndim = len(shape)

    # one-step grid
    if u.ndim == spatial_ndim + 2:
        B = u.shape[0]
        C = u.shape[-1]
        flat = u.reshape(B, P, C)
        dec = norm.decode(flat)
        return dec.reshape(B, *shape, C)

    # rollout grid
    if u.ndim == spatial_ndim + 3:
        B = u.shape[0]
        S = u.shape[1]
        C = u.shape[-1]
        flat = u.reshape(B * S, P, C)
        dec = norm.decode(flat)
        return dec.reshape(B, S, *shape, C)

    return norm.decode(u)


@dataclass
class InferenceResult:
    """Notebook-facing container."""

    split: Optional[str]
    metrics: Dict[str, float]
    rollout_info: Optional[Dict[str, Any]]
    shape: List[int]

    x: Optional[Union[torch.Tensor, np.ndarray]] = None
    y: Optional[Union[torch.Tensor, np.ndarray]] = None
    y_pred: Optional[Union[torch.Tensor, np.ndarray]] = None

    coords: Optional[Union[torch.Tensor, np.ndarray]] = None
    geom: Optional[Dict[str, Any]] = None

    y_denorm: Optional[Union[torch.Tensor, np.ndarray]] = None
    y_pred_denorm: Optional[Union[torch.Tensor, np.ndarray]] = None


class BaseForecaster(object):
    """Lightweight inference / evaluation helper for a trained run.

    - Loads config.yaml and best_model.pth from a run directory.
    - Builds model and evaluator.
    - Provides one-step + rollout inference for unified and grid layouts.
    - Provides notebook-friendly `infer_split` / plotting helpers.
    """

    def __init__(self, path: str, device: Optional[str] = None, data_bundle: Optional[DataBundle] = None) -> None:
        self.saving_path = path

        # ----------------- config -----------------
        args_path = os.path.join(self.saving_path, "config.yaml")
        if not os.path.isfile(args_path):
            raise FileNotFoundError(f"config.yaml not found in {self.saving_path}")
        with open(args_path, "r") as f:
            self.args: Dict[str, Any] = yaml.safe_load(f)

        self.model_args = self.args["model"]
        self.train_args = self.args["train"]
        self.data_args = self.args["data"]

        # spatial shape, e.g. [L], [H, W], [D, H, W]
        self.shape: List[int] = list(self.data_args["shape"])
        self.data_name = self.data_args["name"]

        # ----------------- device & seed -----------------
        seed = self.train_args.get("seed", 42)
        set_seed(seed)

        if device is not None:
            self.device = torch.device(device)
        else:
            default_dev = self.train_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(default_dev)

        # ----------------- geometry placeholders -----------------
        self.geom: Optional[Dict[str, Any]] = None
        self.coords: Optional[torch.Tensor] = None

        # Model forward signature flags
        self._model_accepts_coords: bool = False
        self._model_accepts_geom: bool = False
        self._model_accepts_y: bool = False

        # ----------------- model -----------------
        self.model_name = self.model_args["name"]
        self.model = self.build_model()
        self.load_model()
        self.model.to(self.device)

        self._inspect_model_signature()

        # ----------------- evaluator -----------------
        self.build_evaluator()

        # Normalizers / loaders are built on demand
        self.x_normalizer: Optional[Any] = None
        self.y_normalizer: Optional[Any] = None
        self.train_loader: Optional[data.DataLoader] = None
        self.valid_loader: Optional[data.DataLoader] = None
        self.test_loader: Optional[data.DataLoader] = None
        
        if data_bundle is not None:
            self.attach_data_bundle(data_bundle)

    # ------------------------------------------------------------------
    # Shape utilities (grid vs unified)
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_grid_one_step(y: torch.Tensor, shape: Optional[List[int]]) -> bool:
        # (B, *shape, C)
        if shape is None:
            return False
        if y.ndim != len(shape) + 2:
            return False
        return list(y.shape[1 : 1 + len(shape)]) == list(shape)

    @staticmethod
    def _looks_like_grid_rollout(y: torch.Tensor, shape: Optional[List[int]]) -> bool:
        # (B, S, *shape, C)
        if shape is None:
            return False
        if y.ndim != len(shape) + 3:
            return False
        return list(y.shape[2 : 2 + len(shape)]) == list(shape)

    def _as_grid_one_step(self, u: torch.Tensor) -> torch.Tensor:
        """Convert (B, P, C) -> (B, *shape, C) when P == prod(shape)."""
        if u.ndim == 3:
            P = _prod(self.shape)
            if u.shape[1] == P:
                return u.reshape(u.shape[0], *self.shape, u.shape[-1])
        return u

    def _as_grid_rollout(self, u: torch.Tensor) -> torch.Tensor:
        """Convert (B, S, P, C) -> (B, S, *shape, C) when P == prod(shape)."""
        if u.ndim == 4:
            P = _prod(self.shape)
            if u.shape[2] == P:
                return u.reshape(u.shape[0], u.shape[1], *self.shape, u.shape[-1])
        return u

    # ------------------------------------------------------------------
    # Model / signature
    # ------------------------------------------------------------------
    def build_model(self, **kwargs: Any) -> torch.nn.Module:
        if self.model_name not in MODEL_REGISTRY:
            raise NotImplementedError(f"Model {self.model_name} not implemented")
        print(f"Building model: {self.model_name}")
        model_cls = MODEL_REGISTRY[self.model_name]
        return model_cls(self.model_args)

    def load_model(self, **kwargs: Any) -> None:
        model_path = os.path.join(self.saving_path, "best_model.pth")
        if os.path.isfile(model_path):
            print(f"=> loading checkpoint '{model_path}'")
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"=> no checkpoint found at '{model_path}'")

    def _inspect_model_signature(self) -> None:
        """Record whether model.forward accepts coords / geom / y."""
        try:
            sig = inspect.signature(self.model.forward)
        except (TypeError, ValueError):
            self._model_accepts_coords = False
            self._model_accepts_geom = False
            self._model_accepts_y = False
            return

        params = sig.parameters
        self._model_accepts_coords = "coords" in params
        self._model_accepts_geom = "geom" in params
        self._model_accepts_y = "y" in params

        print(
            f"Model forward supports: coords={self._model_accepts_coords}, "
            f"geom={self._model_accepts_geom}, y={self._model_accepts_y}"
        )

    def _forward_model(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """Unified forward: auto-injects coords / geom / y if supported."""
        kwargs: Dict[str, Any] = dict(extra_kwargs)

        if self._model_accepts_coords and self.coords is not None:
            # Safest default: only inject point-cloud coords when x is (B, P, C) and coords is (P, d).
            coords = self.coords.to(self.device)
            if x.ndim == 3 and coords.dim() == 2 and coords.shape[0] == x.shape[1]:
                B = x.shape[0]
                kwargs["coords"] = coords.unsqueeze(0).expand(B, -1, -1)

        if self._model_accepts_geom and self.geom is not None:
            kwargs["geom"] = self.geom

        if self._model_accepts_y and y is not None:
            kwargs["y"] = y

        out = self.model(x, **kwargs) if len(kwargs) > 0 else self.model(x)

        if isinstance(out, dict):
            out = out.get("pred", out.get("y", None))
            if out is None:
                raise KeyError("Model returned dict but missing keys {'pred','y'} (adapt BaseForecaster._forward_model).")

        return out

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------
    def build_evaluator(self) -> None:
        # Configurable evaluation metrics (from saved config.yaml: `evaluate`).
        eval_args = (self.args.get("evaluate") or {})
        metric_cfg = eval_args.get("metrics", None)
        strict = bool(eval_args.get("strict", True))
        rollout_args = (eval_args.get("rollout") or {})
        rollout_per_step = bool(rollout_args.get("per_step", True))
        metric_kwargs = eval_args.get("metric_kwargs", None) or eval_args.get("kwargs", None) or {}

        self.evaluator = Evaluator(
            shape=self.shape,
            metric_cfg=metric_cfg,
            strict=strict,
            rollout_per_step=rollout_per_step,
            **metric_kwargs,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def build_data(
        self,
        **kwargs: Any,
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, Any, Any]:
        """Build dataset and dataloaders; also captures coords/geom if provided."""
        if self.data_name not in DATASET_REGISTRY:
            raise NotImplementedError(f"Dataset {self.data_name} not implemented")

        dataset_cls = DATASET_REGISTRY[self.data_name]
        dataset = dataset_cls(self.data_args, **kwargs)

        # geometry / coordinates
        self.geom = getattr(dataset, "geom", None)
        self.coords = getattr(dataset, "coords", None)
        if self.coords is not None:
            self.coords = self.coords.to(self.device)
        if self.geom is not None:
            print(f"Dataset geometry: {self.geom}")

        train_loader, valid_loader, test_loader, _ = dataset.make_loaders(
            ddp=False,
            rank=0,
            world_size=1,
            drop_last=True,
        )

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.x_normalizer = dataset.x_normalizer
        self.y_normalizer = dataset.y_normalizer

        return train_loader, valid_loader, test_loader, dataset.x_normalizer, dataset.y_normalizer

    def attach_data_bundle(self, bundle: DataBundle) -> None:
        """
        Attach a pre-built DataBundle to the forecaster.
        Validates data_name and shape consistency.
        Args:
            bundle: DataBundle instance to attach
        """
        if bundle.data_name != self.data_name:
            raise ValueError(f"DataBundle data_name={bundle.data_name} != forecaster data_name={self.data_name}")
        if list(bundle.shape) != list(self.shape):
            raise ValueError(f"DataBundle shape={bundle.shape} != forecaster shape={self.shape}")

        self.train_loader = bundle.train_loader
        self.valid_loader = bundle.valid_loader
        self.test_loader = bundle.test_loader
        self.x_normalizer = bundle.x_normalizer
        self.y_normalizer = bundle.y_normalizer

        # geometry / coordinates
        self.geom = bundle.geom
        self.coords = bundle.coords
        if self.coords is not None:
            self.coords = self.coords.to(self.device)
    
    # ------------------------------------------------------------------
    # Inference (one-step / rollout)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout_inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generic autoregressive rollout.

        Assumes x last dim = static_dim + field_dim, and uses the last `field_dim` as the evolving state.
        Works for both unified and grid layouts.
        """
        if y.ndim < 3:
            raise ValueError(f"rollout_inference expects at least 3D y, got shape={tuple(y.shape)}")

        steps = int(y.shape[1])
        in_dim = x.shape[-1]
        field_dim = y.shape[-1]

        if in_dim < field_dim:
            raise ValueError(
                f"Input last dim {in_dim} < target field dim {field_dim}. Cannot separate static and dynamic channels."
            )

        static_dim = in_dim - field_dim
        if static_dim > 0:
            static = x[..., :static_dim].contiguous()
            cur = x[..., static_dim:].contiguous()
        else:
            static = None
            cur = x

        preds: List[torch.Tensor] = []
        self.model.eval()

        for s in range(steps):
            xin = torch.cat([static, cur], dim=-1) if static is not None else cur
            nxt = self._forward_model(xin, y=None, **kwargs)
            nxt = nxt.reshape(y[:, s].shape)
            preds.append(nxt)
            cur = nxt

        return torch.stack(preds, dim=1)  # (B, S, ..., field_dim)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Dispatch based on target dimensionality (grid or unified).

        One-step:
          - unified: (B, P, C)
          - grid:    (B, *shape, C)

        Rollout:
          - unified: (B, S, P, C)
          - grid:    (B, S, *shape, C)
        """
        # rollout grid
        if self._looks_like_grid_rollout(y, self.shape):
            return self.rollout_inference(x, y, **kwargs)

        # rollout unified (ambiguous with 2D grid one-step)
        if y.ndim == 4 and not self._looks_like_grid_one_step(y, self.shape):
            return self.rollout_inference(x, y, **kwargs)

        # one-step unified
        if y.ndim == 3:
            out = self._forward_model(x, y=None, **kwargs)
            return out.reshape(y.shape)

        # one-step grid
        if self._looks_like_grid_one_step(y, self.shape):
            out = self._forward_model(x, y=None, **kwargs)
            return out.reshape(y.shape)

        raise ValueError(f"Unsupported y.ndim={y.ndim}, y.shape={tuple(y.shape)}, configured shape={self.shape}")

    # ------------------------------------------------------------------
    # Notebook-friendly inference
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch.get("x", None)
            y = batch.get("y", None)
            if x is None or y is None:
                raise KeyError("Batch dict must contain keys 'x' and 'y' (or adjust BaseForecaster._unpack_batch).")
            return x, y
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Batch tuple/list must be (x, y) or (x, y, ...).")
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    @torch.no_grad()
    def infer_loader(
        self,
        loader: data.DataLoader,
        *,
        split: Optional[str] = None,
        max_batches: Optional[int] = None,
        max_samples: Optional[int] = None,
        store_inputs: bool = False,
        store_outputs: bool = True,
        to_cpu: bool = True,
        to_numpy: bool = False,
        denormalize: bool = False,
        return_rollout_info: bool = True,
        verbose: bool = False,
    ) -> InferenceResult:
        self.model.eval()
        loss_record = self.evaluator.init_record()

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        preds: List[torch.Tensor] = []

        rollout_steps: Optional[int] = None
        per_step_sum: Dict[str, np.ndarray] = {}
        per_step_weight: float = 0.0

        seen = 0

        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break

            x, y = self._unpack_batch(batch)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            bsz = int(x.shape[0])
            if max_samples is not None:
                if seen >= max_samples:
                    break
                if seen + bsz > max_samples:
                    keep = max_samples - seen
                    x = x[:keep]
                    y = y[:keep]
                    bsz = int(keep)

            y_pred = self.inference(x, y)

            metrics_out = self.evaluator(y_pred, y, record=loss_record, batch_size=bsz)

            # rollout curves aggregation (robust: scan keys in metrics_out)
            if return_rollout_info and isinstance(metrics_out, dict) and "rollout_steps" in metrics_out:
                rollout_steps = int(metrics_out["rollout_steps"])
                per_step_weight += float(bsz)

                for k, v in metrics_out.items():
                    if not isinstance(k, str) or not k.endswith("_per_step"):
                        continue
                    if v is None:
                        continue

                    base = k[:-len("_per_step")]  # e.g. "rmse"
                    arr_np = np.asarray(v, dtype=np.float64)

                    if base not in per_step_sum:
                        per_step_sum[base] = np.zeros_like(arr_np)
                    per_step_sum[base] += arr_np * float(bsz)

            if store_outputs:
                if to_cpu:
                    ys.append(y.detach().cpu())
                    preds.append(y_pred.detach().cpu())
                else:
                    ys.append(y.detach())
                    preds.append(y_pred.detach())

            if store_inputs:
                if to_cpu:
                    xs.append(x.detach().cpu())
                else:
                    xs.append(x.detach())

            seen += bsz

        metrics_scalar: Dict[str, float] = {}
        for k in loss_record.loss_list:
            if k not in loss_record.loss_dict:
                continue
            v = float(loss_record.loss_dict[k].avg)
            metrics_scalar[k] = int(round(v)) if k == "rollout_steps" else v

        if verbose:
            print(loss_record)

        x_all: Optional[torch.Tensor] = torch.cat(xs, dim=0) if (store_inputs and len(xs) > 0) else None
        y_all: Optional[torch.Tensor] = torch.cat(ys, dim=0) if (store_outputs and len(ys) > 0) else None
        y_pred_all: Optional[torch.Tensor] = torch.cat(preds, dim=0) if (store_outputs and len(preds) > 0) else None

        rollout_info: Optional[Dict[str, Any]] = None
        if return_rollout_info and rollout_steps is not None:
            rollout_info = {"rollout_steps": int(rollout_steps)}
            if per_step_weight > 0:
                for key, ssum in per_step_sum.items():
                    mean_arr = (ssum / per_step_weight).tolist()
                    rollout_info[f"{key}_per_step"] = mean_arr
                    rollout_info[f"{key}_rollout_mean"] = float(np.mean(mean_arr))

        y_denorm: Optional[torch.Tensor] = None
        y_pred_denorm: Optional[torch.Tensor] = None
        if denormalize and self.y_normalizer is not None and y_all is not None and y_pred_all is not None:
            y_denorm = _decode_field(self.y_normalizer, y_all, shape=self.shape).detach().cpu()
            y_pred_denorm = _decode_field(self.y_normalizer, y_pred_all, shape=self.shape).detach().cpu()

        def _maybe_np(t: Optional[torch.Tensor]) -> Optional[Union[torch.Tensor, np.ndarray]]:
            if t is None:
                return None
            if to_numpy:
                return t.detach().cpu().numpy()
            return t

        coords_out: Optional[Union[torch.Tensor, np.ndarray]] = None
        if self.coords is not None:
            coords_out = self.coords.detach().cpu().numpy() if to_numpy else self.coords

        return InferenceResult(
            split=split,
            metrics=metrics_scalar,
            rollout_info=rollout_info,
            shape=list(self.shape),
            x=_maybe_np(x_all),
            y=_maybe_np(y_all),
            y_pred=_maybe_np(y_pred_all),
            coords=coords_out,
            geom=self.geom,
            y_denorm=_maybe_np(y_denorm) if isinstance(y_denorm, torch.Tensor) else y_denorm,
            y_pred_denorm=_maybe_np(y_pred_denorm) if isinstance(y_pred_denorm, torch.Tensor) else y_pred_denorm,
        )

    def infer_split(
        self,
        split: str = "test",
        *,
        max_batches: Optional[int] = None,
        max_samples: Optional[int] = None,
        store_inputs: bool = False,
        store_outputs: bool = True,
        to_cpu: bool = True,
        to_numpy: bool = False,
        denormalize: bool = False,
        return_rollout_info: bool = True,
        verbose: bool = False,
        **data_kwargs: Any,
    ) -> InferenceResult:
        if self.test_loader is None or self.valid_loader is None or self.train_loader is None:
            self.build_data(**data_kwargs)

        if split == "train":
            loader = self.train_loader
        elif split in ("val", "valid", "validation"):
            loader = self.valid_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown split='{split}', expected one of: train/val/test.")

        assert loader is not None
        return self.infer_loader(
            loader,
            split=split,
            max_batches=max_batches,
            max_samples=max_samples,
            store_inputs=store_inputs,
            store_outputs=store_outputs,
            to_cpu=to_cpu,
            to_numpy=to_numpy,
            denormalize=denormalize,
            return_rollout_info=return_rollout_info,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Backward-compatible evaluation API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        loader: data.DataLoader,
        *,
        return_rollout_info: bool = False,
        verbose: bool = False,
    ) -> InferenceResult:
        res = self.infer_loader(
            loader,
            split=None,
            store_inputs=False,
            store_outputs=False,
            to_cpu=False,
            to_numpy=False,
            denormalize=False,
            return_rollout_info=return_rollout_info,
            verbose=verbose,
        )
        return res

    # ------------------------------------------------------------------
    # Minimal visualization helpers (for notebooks)
    # ------------------------------------------------------------------
    def plot_rollout_curves(
        self,
        result: InferenceResult,
        *,
        keys: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ):
        """Plot rollout per-step curves if available."""
        import matplotlib.pyplot as plt

        info = result.rollout_info
        if info is None:
            raise ValueError("No rollout_info in result. Run infer_split(..., return_rollout_info=True) on rollout data.")

        steps = int(info.get("rollout_steps", 0))
        if steps <= 0:
            raise ValueError(f"Invalid rollout_steps={steps}")

        if keys is None:
            keys = sorted([k.replace("_per_step", "") for k in info.keys() if k.endswith("_per_step")])

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        xs = np.arange(steps)
        for k in keys:
            ys = info.get(f"{k}_per_step", None)
            if ys is None:
                continue
            ax.plot(xs, ys, label=k)
        ax.set_xlabel("Step")
        ax.set_ylabel("Metric")
        ax.legend()
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        return fig

    def _select_step_tensor(self, t: torch.Tensor, step: int) -> torch.Tensor:
        """
        If t is rollout:
        - unified: (B,S,N,C) -> (B,N,C)
        - grid:    (B,S,H,W,C) -> (B,H,W,C)
        else return t.
        """
        if t.ndim == 4 and not self._looks_like_grid_one_step(t, self.shape):
            # (B,S,N,C)
            return t[:, step]
        if t.ndim == 5:
            # (B,S,H,W,C)
            return t[:, step]
        return t

    def _to_hw(self, t: torch.Tensor, idx: int, channel: int) -> torch.Tensor:
        """
        Convert one-step tensor to (H,W) for 2D shape.
        Accepts:
        - unified: (B,N,C) -> (H,W)
        - grid:    (B,H,W,C) -> (H,W)
        """
        if len(self.shape) != 2:
            raise ValueError(f"_to_hw requires 2D shape, got shape={self.shape}")

        H, W = self.shape
        if t.ndim == 3:
            # (B,N,C)
            B, N, C = t.shape
            if N != H * W:
                raise ValueError(f"Cannot reshape N={N} to (H,W)=({H},{W})")
            ch = min(channel, C - 1)
            return t[idx, :, ch].reshape(H, W)

        if t.ndim == 4:
            # (B,H,W,C)
            B, h, w, C = t.shape
            if [h, w] != [H, W]:
                raise ValueError(f"Grid mismatch: got (h,w)=({h},{w}), expected ({H},{W})")
            ch = min(channel, C - 1)
            return t[idx, :, :, ch]

        raise ValueError(f"Unsupported tensor for 2D frame extraction: {tuple(t.shape)}")

    def vis_sample(
        self,
        result: "InferenceResult",
        *,
        idx: int = 0,
        step: int = 0,
        x_channel: int = 0,
        y_channel: int = 0,
        use_denorm: bool = False,
        save_path: Optional[str] = None,
        **vis_kwargs: Any,
    ) -> None:
        """
        Dataset-specific visualization via utils.vis registry.

        Requirements:
        - result must have stored y and y_pred
        - if you want input visualization, result must have stored x (store_inputs=True)

        For NS2D, it calls ns2d_vis(raw_x, raw_y, pred_y, ...).
        """
        fn = get_vis_fn(self.data_name)
        if fn is None:
            raise ValueError(
                f"No visualization function registered for dataset '{self.data_name}'. "
                f"Register it in utils/vis.py::VIS_REGISTRY."
            )

        # pick tensors (denorm if requested & available)
        y = result.y_denorm if (use_denorm and result.y_denorm is not None) else result.y
        yp = result.y_pred_denorm if (use_denorm and result.y_pred_denorm is not None) else result.y_pred
        x = result.x

        if y is None or yp is None:
            raise ValueError("Result missing y/y_pred. Run infer_split(..., store_outputs=True).")
        if x is None:
            raise ValueError("Result missing x. Run infer_split(..., store_inputs=True) if your vis needs inputs.")

        # ensure torch tensors
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        else:
            x_t = x
        if isinstance(y, np.ndarray):
            y_t = torch.from_numpy(y)
        else:
            y_t = y
        if isinstance(yp, np.ndarray):
            yp_t = torch.from_numpy(yp)
        else:
            yp_t = yp

        # select step if rollout
        x_t = self._select_step_tensor(x_t, step)
        y_t = self._select_step_tensor(y_t, step)
        yp_t = self._select_step_tensor(yp_t, step)

        # extract 2D frames (H,W)
        raw_x = self._to_hw(x_t, idx=idx, channel=x_channel)
        raw_y = self._to_hw(y_t, idx=idx, channel=y_channel)
        pred_y = self._to_hw(yp_t, idx=idx, channel=y_channel)

        # call dataset-specific function
        fn(raw_x, raw_y, pred_y, save_path=save_path, **vis_kwargs)
