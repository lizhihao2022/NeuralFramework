# forecastors/base.py
from __future__ import annotations

import os
import inspect
from typing import Any, Dict, Optional, List, Tuple

import yaml
import torch
import torch.utils.data as data

from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from utils import Evaluator, set_seed


def _decode_field(
    norm: Optional[Any],
    u: torch.Tensor,
    shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Decode a field tensor using a normalizer that was fit on (B, P, C).

    shape:
        Spatial shape used when the normalizer was fit, e.g.
        [L], [H, W], [D, H, W]. If None, fall back to norm.decode(u).
    """
    if norm is None or not hasattr(norm, "decode"):
        return u

    if shape is None:
        return norm.decode(u)

    spatial_ndim = len(shape)

    # one-step: (B, *shape, C)
    if u.ndim == spatial_ndim + 2:
        B = u.shape[0]
        C = u.shape[-1]
        P = 1
        for s in shape:
            P *= s
        flat = u.reshape(B, P, C)
        dec = norm.decode(flat)
        return dec.reshape(B, *shape, C)

    # rollout: (B, S, *shape, C)
    if u.ndim == spatial_ndim + 3:
        B = u.shape[0]
        S = u.shape[1]
        C = u.shape[-1]
        P = 1
        for s in shape:
            P *= s
        flat = u.reshape(B * S, P, C)
        dec = norm.decode(flat)
        return dec.reshape(B, S, *shape, C)

    # Fallback: let the normalizer handle it directly
    return norm.decode(u)


class BaseForecaster(object):
    """
    Lightweight inference / evaluation helper for a trained run.

    - Loads config.yaml and best_model.pth from a run directory.
    - Builds model and evaluator.
    - Provides one-step and rollout forecast utilities.

    Geometry-aware:
      - If the dataset exposes `coords` (e.g. (N, d)) and/or `geom` dict,
        they are stored and, when supported by the model, automatically
        passed as `coords=...` / `geom=...` to model.forward().
    """

    def __init__(self, path: str, device: Optional[str] = None) -> None:
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
            default_dev = self.train_args.get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
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

        # Inspect forward signature once
        self._inspect_model_signature()

        # ----------------- evaluator -----------------
        self.build_evaluator()

        # Normalizers / loaders are built on demand
        self.x_normalizer: Optional[Any] = None
        self.y_normalizer: Optional[Any] = None
        self.train_loader: Optional[data.DataLoader] = None
        self.valid_loader: Optional[data.DataLoader] = None
        self.test_loader: Optional[data.DataLoader] = None

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
        """
        Record whether model.forward accepts coords / geom / y.
        """
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
            f"Model forward supports: "
            f"coords={self._model_accepts_coords}, "
            f"geom={self._model_accepts_geom}, "
            f"y={self._model_accepts_y}"
        )

    def _forward_model(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """
        Unified forward:
          - auto-injects coords / geom / y if model supports them.
        """
        kwargs: Dict[str, Any] = dict(extra_kwargs)

        if self._model_accepts_coords and self.coords is not None:
            B = x.shape[0]
            coords = self.coords.to(self.device)
            if coords.dim() == 2:
                coords_batched = coords.unsqueeze(0).expand(B, -1, -1)
            else:
                coords_batched = coords
            kwargs["coords"] = coords_batched

        if self._model_accepts_geom and self.geom is not None:
            kwargs["geom"] = self.geom

        if self._model_accepts_y and (y is not None):
            kwargs["y"] = y

        return self.model(x, **kwargs)

    # ------------------------------------------------------------------
    # Evaluator / dataset
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

    def build_data(
        self,
        **kwargs: Any,
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, Any, Any]:
        """
        Build dataset and dataloaders.

        Also captures dataset.coords / dataset.geom if available.
        """
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

        # store for convenience
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.x_normalizer = dataset.x_normalizer
        self.y_normalizer = dataset.y_normalizer

        return train_loader, valid_loader, test_loader, dataset.x_normalizer, dataset.y_normalizer

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
        """
        Generic autoregressive rollout.

        Assumes the last channel of x corresponds to the evolving field,
        and any leading channels are static features (e.g. coordinates).
        For each step:
            - concatenate static features (if any) with current field
            - call model
            - update current field
        Shapes:
            x: (B, ..., C_in)
            y: (B, S, ..., C_out)
        """
        if y.ndim < 3:
            raise ValueError(f"rollout_inference expects at least 3D y, got shape={tuple(y.shape)}")

        steps = int(y.shape[1])
        in_dim = x.shape[-1]
        field_dim = y.shape[-1]

        if in_dim < field_dim:
            raise ValueError(
                f"Input last dim {in_dim} < target field dim {field_dim}. "
                "Cannot separate static and dynamic channels."
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
            if static is not None:
                xin = torch.cat([static, cur], dim=-1)
            else:
                xin = cur

            nxt = self._forward_model(xin, y=None, **kwargs)

            # match the per-step target shape
            target_shape = y[:, s].shape
            nxt = nxt.reshape(target_shape)

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
        """
        Dispatch based on target dimensionality:

        - one-step: y shape (B, *shape, C)      -> model(x) reshaped to y
        - rollout:  y shape (B, S, *shape, C)   -> autoregressive rollout
        """
        # one-step
        if y.ndim == len(self.shape) + 2:
            out = self._forward_model(x, y=None, **kwargs)
            return out.reshape(y.shape)

        # rollout
        if y.ndim == len(self.shape) + 3:
            return self.rollout_inference(x, y, **kwargs)

        raise ValueError(f"Unsupported y.ndim={y.ndim} with shape={tuple(y.shape)}")

    # ------------------------------------------------------------------
    # Forecast / metrics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self,
        loader,
        *,
        return_rollout_info: bool = False,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Run inference on `loader` and compute metrics.

        This version unifies:
        - one-step: pred/target in (B, N, C) or (B, *shape, C)
        - rollout:  pred/target in (B, S, N, C) or (B, S, *shape, C)

        Returns:
        metrics_scalar: dict of scalar metrics averaged by LossRecord (e.g., rmse/psnr/ssim or *_rollout_mean)
        rollout_info:   optional dict containing per-step curves for rollout, e.g.:
                        {"rollout_steps": S, "rmse_per_step": [...], "rmse_rollout_mean": ...}
                        Actually keys follow your config metric keys:
                        - "<metric>_per_step"
                        - "<metric>_rollout_mean"
        """
        assert hasattr(self, "model"), "BaseForecaster must have self.model"
        assert hasattr(self, "evaluator"), "Call build_evaluator() before forecast()."

        self.model.eval()

        # Create a LossRecord with metric keys (plus optional extra keys if you want)
        loss_record = self.evaluator.init_record()

        all_y: List[torch.Tensor] = []
        all_y_pred: List[torch.Tensor] = []

        # ----------------- inference loop -----------------
        for batch in loader:
            # --- adapt this block to your dataset's batch format ---
            if isinstance(batch, dict):
                x = batch.get("x", None)
                y = batch.get("y", None)
                if x is None or y is None:
                    raise KeyError("Batch dict must contain keys 'x' and 'y' (or adjust this code).")
            elif isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Batch tuple/list must be (x, y) or (x, y, ...).")
                x, y = batch[0], batch[1]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # --- forward ---
            # If your model returns dict, adapt here.
            y_pred = self.model(x)
            if isinstance(y_pred, dict):
                # common patterns: {"pred": ...} / {"y": ...}
                y_pred = y_pred.get("pred", y_pred.get("y", None))
                if y_pred is None:
                    raise KeyError("Model output dict must contain 'pred' or 'y' (or adjust this code).")

            # Keep on GPU for now; concatenate later
            all_y.append(y)
            all_y_pred.append(y_pred)

        if len(all_y) == 0:
            raise ValueError("Empty loader: no batches to forecast.")

        y_all = torch.cat(all_y, dim=0)
        y_pred_all = torch.cat(all_y_pred, dim=0)

        # ----------------- metrics (unified) -----------------
        # Evaluator will:
        #   - detect rollout vs one-step
        #   - compute scalar metrics and (optionally) per-step curves for rollout
        metrics_out = self.evaluator(y_pred_all, y_all, record=loss_record)

        rollout_info: Optional[Dict[str, Any]] = None
        if return_rollout_info and isinstance(metrics_out, dict) and "rollout_steps" in metrics_out:
            # Only keep rollout-related fields (do NOT try to put lists into LossRecord)
            rollout_info = {
                k: v
                for k, v in metrics_out.items()
                if k == "rollout_steps" or k.endswith("_per_step") or k.endswith("_rollout_mean")
            }

        # Convert LossRecord averages into a clean scalar dict for return/logging
        metrics_scalar: Dict[str, float] = {}
        for k in loss_record.loss_list:
            if k in loss_record.loss_dict:
                metrics_scalar[k] = float(loss_record.loss_dict[k].avg)

        # (optional) print for debugging
        print(loss_record)

        return metrics_scalar, rollout_info
