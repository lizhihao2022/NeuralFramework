# datasets/ns_2d.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import os.path as osp
import numpy as np
import torch
import scipy.io as sio
from h5py import File, Dataset as H5Dataset

from .base import BaseDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class NS2DDataset(BaseDataset):
    """
    2D Navier–Stokes dataset.

    Raw file must contain key 'u':
      - .mat: raw_data['u'] shape (N, H, W, T)
      - .h5:  one of (H, W, T, N) or (T, H, W, N) or (N, H, W, T)

    After loading, we standardize to (N, H, W, T).

    Tasks (data_args['task']):

      - "one_step":
          all splits (train/valid/test) use one-step pairs
          x: (B, N_points, 1), y: (B, N_points, 1).

      - "rollout":
          train split still uses one-step pairs (same as above);
          valid/test can use rollout targets:
            x: (B, N_points, 1)
            y: (B, S, N_points, 1)
          where S = rollout_steps.

    Geometry:

      - self.geom: {"dim": 2, "layout": "grid", "spatial_shape": (H', W'), "sample_factor": sf}
      - self.coords: (H'*W', 2) in [0, 1]^2, order matches flatten layout.
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        self.sample_factor: int = int(data_args.get("sample_factor", 1))
        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))

        self.in_t: int = int(data_args.get("in_t", 5))
        self.out_t: int = int(data_args.get("out_t", 1))
        self.duration: int = int(data_args.get("duration", 10))

        self.task: str = str(data_args.get("task", "one_step"))  # "one_step" | "rollout"

        self.rollout_steps: int = int(data_args.get("rollout_steps", self.duration))
        self.rollout_stride: int = int(data_args.get("rollout_stride", 0))

        super().__init__(data_args, **kwargs)

    def get_cache_path(self) -> str:
        if not self.data_path:
            return "ns2d_processed.pt"

        root, _ = osp.splitext(self.data_path)
        sf = self.sample_factor
        norm_tag = self.normalizer_type if self.normalize else "none"

        task_tag = self.task
        if task_tag == "rollout":
            extra = f"_ro{self.rollout_steps}_rs{self.rollout_stride}"
        else:
            extra = ""

        return (
            f"{root}_sf{sf}"
            f"_it{self.in_t}_ot{self.out_t}_dur{self.duration}"
            f"{extra}_task{task_tag}_norm{norm_tag}_processed.pt"
        )

    def load_raw_data(self, **kwargs: Any) -> torch.Tensor:
        data_path = self.data_path
        if not data_path:
            raise ValueError("data_path is empty.")

        # try .mat
        try:
            raw_data = sio.loadmat(data_path)
            if "u" not in raw_data:
                raise KeyError("Key 'u' not found in .mat file.")
            data = torch.tensor(raw_data["u"], dtype=torch.float32)  # (N,H,W,T)
            if data.ndim != 4:
                raise ValueError(f"Expected 'u' to be 4D in .mat, got {tuple(data.shape)}.")
            return data
        except Exception:
            pass

        # fall back to .h5
        with File(data_path, "r") as raw_data:
            u_node = raw_data.get("u", None)
            if u_node is None:
                raise KeyError("Key 'u' not found in HDF5 file.")
            if not isinstance(u_node, H5Dataset):
                raise TypeError("Expected dataset 'u' in HDF5 file.")
            u_array = np.asarray(u_node[()])

        if u_array.ndim != 4:
            raise ValueError(f"Expected 'u' to be 4D in HDF5, got shape={u_array.shape}.")

        d0, d1, d2, d3 = u_array.shape

        def _is_hw(x: int) -> bool:
            return x in (16, 32, 48, 64, 96, 128, 256) or x >= 16

        def _is_t(x: int) -> bool:
            return 1 <= x <= 256

        def _is_n(x: int) -> bool:
            return x >= 64

        if _is_hw(d0) and _is_hw(d1) and _is_t(d2) and _is_n(d3):
            std = np.transpose(u_array, (3, 0, 1, 2))  # (H,W,T,N)->(N,H,W,T)
        elif _is_t(d0) and _is_hw(d1) and _is_hw(d2) and _is_n(d3):
            std = np.transpose(u_array, (3, 1, 2, 0))  # (T,H,W,N)->(N,H,W,T)
        elif _is_n(d0) and _is_hw(d1) and _is_hw(d2) and _is_t(d3):
            std = u_array  # already (N,H,W,T)
        else:
            raise ValueError(
                f"Unrecognized HDF5 layout for 'u' with shape={u_array.shape}. "
                "Expected one of (H,W,T,N), (T,H,W,N), (N,H,W,T)."
            )

        return torch.tensor(std, dtype=torch.float32)

    def process_split(
        self,
        data_split: torch.Tensor,
        mode: str,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any], Optional[Any]]:
        """
        Args:
            data_split: (N, H, W, T)
            mode: 'train' | 'valid' | 'test'

        Returns:
            train (any task): x: (B, N_points, 1), y: (B, N_points, 1)
            valid/test, task="one_step":
                x: (B, N_points, 1), y: (B, N_points, 1)
            valid/test, task="rollout":
                x: (B, N_points, 1), y: (B, S, N_points, 1)
        """
        if data_split.ndim != 4:
            raise ValueError(f"Expected (N,H,W,T), got {tuple(data_split.shape)}.")
        if self.out_t <= 0:
            raise ValueError(f"out_t must be > 0, got {self.out_t}.")

        in_t = self.in_t
        out_t = self.out_t
        duration = self.duration

        if in_t < 0:
            raise ValueError(f"in_t must be >= 0, got {in_t}.")
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}.")

        # train always uses one-step; rollout only used for valid/test
        effective_task = "one_step" if mode == "train" or mode == "valid" else self.task

        # --- temporal windows ---
        T = int(data_split.shape[-1])

        if effective_task == "one_step":
            end_t_task = in_t + duration + out_t
        else:  # rollout for valid/test
            steps = int(self.rollout_steps)
            if steps <= 0:
                raise ValueError(f"rollout_steps must be > 0, got {steps}.")
            end_t_task = in_t + steps * out_t + 1

        if end_t_task > T:
            raise ValueError(f"Temporal window exceeds T: end_t_task={end_t_task} > T={T}.")

        end_t_norm = in_t + duration + out_t
        if end_t_norm > T:
            end_t_norm = end_t_task

        # normalizer window: (N,H,W,Tn)->(N,Tn,H,W)
        data_norm = data_split[..., in_t:end_t_norm].permute(0, 3, 1, 2)
        # task window: (N,H,W,Tt)->(N,Tt,H,W)
        data_task = data_split[..., in_t:end_t_task].permute(0, 3, 1, 2)

        # spatial subsampling
        sf = self.sample_factor
        if sf > 1:
            data_norm = data_norm[:, :, ::sf, ::sf]
            data_task = data_task[:, :, ::sf, ::sf]

        n, t_norm, h, w = data_norm.shape
        _, t_task, _, _ = data_task.shape

        # --- normalization ---
        if self.normalize:
            flat_norm = data_norm.reshape(n * t_norm, -1, 1)

            if mode == "train":
                if self.normalizer_type == "PGN":
                    x_normalizer = UnitGaussianNormalizer(flat_norm)
                    y_normalizer = x_normalizer
                else:
                    x_normalizer = GaussianNormalizer(flat_norm)
                    y_normalizer = x_normalizer
            else:
                if x_normalizer is None or y_normalizer is None:
                    raise RuntimeError("Normalizer is None for non-train split.")

            flat_task = data_task.reshape(n * t_task, -1, 1)
            data_task = x_normalizer.encode(flat_task).reshape(n, t_task, h, w)

        # --- geometry / coords (once) ---
        if self.geom is None or self.coords is None:
            self.geom = {
                "dim": 2,
                "layout": "grid",
                "spatial_shape": (h, w),
                "sample_factor": sf,
            }
            yy = torch.linspace(0.0, 1.0, h, dtype=torch.float32)
            xx = torch.linspace(0.0, 1.0, w, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
            coords = torch.stack((grid_y, grid_x), dim=-1).view(-1, 2)
            self.coords = coords

        n_points = h * w

        # --- one-step branch (train for all tasks, eval if task="one_step") ---
        if effective_task == "one_step":
            if t_task < duration + out_t:
                raise ValueError(
                    f"Not enough temporal length for one_step: "
                    f"t_task={t_task}, duration={duration}, out_t={out_t}."
                )

            x_u = data_task[:, :duration, :, :]              # (N, duration, H', W')
            y_u = data_task[:, out_t:out_t + duration, :, :] # (N, duration, H', W')

            x_u = x_u.flatten(0, 1)                          # (B, H', W')
            y_u = y_u.flatten(0, 1)                          # (B, H', W')

            B = int(x_u.shape[0])

            x = x_u.view(B, n_points, 1)
            y = y_u.view(B, n_points, 1)

            return x, y, x_normalizer, y_normalizer

        # --- rollout branch (only valid/test when task="rollout") ---
        steps = int(self.rollout_steps)
        max_start = (t_task - 1) - steps * out_t
        if max_start < 0:
            raise ValueError(
                f"Not enough temporal length for rollout: "
                f"t_task={t_task}, steps={steps}, out_t={out_t}."
            )

        if self.rollout_stride and self.rollout_stride > 0:
            starts = list(range(0, max_start + 1, self.rollout_stride))
        else:
            starts = [0]

        xs = []
        ys = []
        for s in starts:
            u0 = data_task[:, s, :, :]  # (N, H', W')
            gt = data_task[:, s + out_t : s + (steps + 1) * out_t : out_t, :, :]  # (N, steps, H', W')
            xs.append(u0)
            ys.append(gt)

        x_seq = torch.cat(xs, dim=0)  # (B', H', W')
        y_seq = torch.cat(ys, dim=0)  # (B', steps, H', W')

        B_total = int(x_seq.shape[0])

        x = x_seq.view(B_total, n_points, 1)
        y = y_seq.view(B_total, steps, n_points, 1)

        return x, y, x_normalizer, y_normalizer
