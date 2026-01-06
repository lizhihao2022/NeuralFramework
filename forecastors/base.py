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
        self.evaluator = Evaluator(self.shape)

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
        loader: data.DataLoader,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        return_all: bool = False,
        return_rollout_info: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run forecast over a dataloader and compute metrics.

        If return_rollout_info=True for rollout targets, also return:
            {
              "rollout_steps": int,
              "rmse_per_step": List[float],
              "rmse_rollout_mean": float
            }
        """
        loss_record = self.evaluator.init_record()

        all_x: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []
        all_y_pred: List[torch.Tensor] = []

        self.model.eval()
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            y_pred = self.inference(x, y, **kwargs)

            # decode y / y_pred to physical scale
            y_pred_phys = _decode_field(y_normalizer, y_pred, self.shape)
            y_phys = _decode_field(y_normalizer, y, self.shape)

            # decode the input field channel (last channel of x)
            if x_normalizer is not None and hasattr(x_normalizer, "decode"):
                x_u = x[..., -1:].contiguous()
                x_u_phys = _decode_field(x_normalizer, x_u, self.shape)
            else:
                x_u_phys = x[..., -1:].contiguous()

            all_x.append(x_u_phys.detach().cpu())
            all_y.append(y_phys.detach().cpu())
            all_y_pred.append(y_pred_phys.detach().cpu())

        x_all = torch.cat(all_x, dim=0)
        y_all = torch.cat(all_y, dim=0)
        y_pred_all = torch.cat(all_y_pred, dim=0)

        rollout_info: Optional[Dict[str, Any]] = None
        spatial_ndim = len(self.shape)

        # ----------------- metrics -----------------
        # one-step: (B, *shape, C)
        if y_all.ndim == spatial_ndim + 2:
            self.evaluator(y_pred_all, y_all, record=loss_record)

        # rollout: (B, S, *shape, C)
        elif y_all.ndim == spatial_ndim + 3:
            B = y_all.shape[0]
            S = y_all.shape[1]
            C = y_all.shape[-1]

            # reshape to (B*S, *shape, C) for Evaluator
            new_shape = (B * S, *self.shape, C)
            y_flat = y_all.reshape(new_shape)
            y_pred_flat = y_pred_all.reshape(new_shape)

            self.evaluator(y_pred_flat, y_flat, record=loss_record)

            # per-step RMSE on physical scale
            diff = (y_pred_all - y_all) ** 2
            reduce_dims = tuple(d for d in range(diff.ndim) if d != 1)  # keep step dim
            mse_per_step = diff.mean(dim=reduce_dims)  # (S,)
            rmse_per_step = torch.sqrt(mse_per_step.clamp_min(0.0))

            rollout_info = {
                "rollout_steps": int(S),
                "rmse_per_step": [float(v) for v in rmse_per_step.tolist()],
                "rmse_rollout_mean": float(rmse_per_step.mean().item()),
            }

        else:
            raise ValueError(
                f"Unexpected target shape {tuple(y_all.shape)} "
                f"for spatial_ndim={spatial_ndim}"
            )

        print(loss_record)

        # ----------------- return -----------------
        if return_all and return_rollout_info:
            return loss_record, x_all, y_all, y_pred_all, rollout_info

        if return_all:
            return loss_record, x_all, y_all, y_pred_all

        if return_rollout_info:
            return loss_record, rollout_info

        return loss_record
