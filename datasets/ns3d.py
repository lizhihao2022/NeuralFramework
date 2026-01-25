# datasets/ns3d.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence
import os.path as osp
import itertools

import numpy as np
import torch
import scipy.io as sio
from h5py import File, Dataset as H5Dataset

from .base import BaseDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


def _as_3tuple(sample_factor: Any) -> Tuple[int, int, int]:
    """Accept int / list / tuple for sample_factor and return a 3-tuple."""
    if sample_factor is None:
        return (1, 1, 1)
    if isinstance(sample_factor, int):
        sf = int(sample_factor)
        return (sf, sf, sf)
    if isinstance(sample_factor, (list, tuple)):
        if len(sample_factor) != 3:
            raise ValueError(f"sample_factor must have length 3, got {sample_factor}.")
        return (int(sample_factor[0]), int(sample_factor[1]), int(sample_factor[2]))
    raise TypeError(f"Unsupported type for sample_factor: {type(sample_factor)}")


def _first_existing_key(h5: Any, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in h5:
            return k
    return None


def _standardize_to_n_t_xyz(
    arr: np.ndarray,
    *,
    x_len: Optional[int],
    y_len: Optional[int],
    z_len: Optional[int],
    t_len: Optional[int],
) -> np.ndarray:
    """
    Attempt to permute a 5D array into (N, T, X, Y, Z) using coordinate lengths.
    If inference fails, fall back to assuming already (N, T, X, Y, Z).
    """
    if arr.ndim != 5:
        raise ValueError(f"Expected 5D array, got shape={arr.shape}.")

    shape = arr.shape
    if any(v is None for v in (x_len, y_len, z_len, t_len)):
        return arr

    x_len = int(x_len)  # type: ignore[assignment]
    y_len = int(y_len)  # type: ignore[assignment]
    z_len = int(z_len)  # type: ignore[assignment]
    t_len = int(t_len)  # type: ignore[assignment]

    candidates = []
    for perm in itertools.permutations(range(5), 5):
        aN, aT, aX, aY, aZ = perm
        if shape[aT] != t_len:
            continue
        if shape[aX] != x_len or shape[aY] != y_len or shape[aZ] != z_len:
            continue

        N = shape[aN]
        score = 0
        if N >= 16:
            score += 2
        if N >= t_len:
            score += 1
        if aN == 0:
            score += 1
        candidates.append((score, perm))

    if not candidates:
        return arr

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, (aN, aT, aX, aY, aZ) = candidates[0]
    return np.transpose(arr, (aN, aT, aX, aY, aZ))


class NS3DDataset(BaseDataset):
    """
    3D Navier–Stokes dataset (grid-based).

    Expected raw HDF5 keys (typical):
      - variables: 'Vx', 'Vy', 'Vz', 'density', 'pressure' (select via data_args['prop'])
      - coordinates: 'x-coordinate', 'y-coordinate', 'z-coordinate', 't-coordinate' (optional)

    After loading, we standardize variable tensor to (N, T, X, Y, Z).

    Tasks (data_args['task']) follow NS2D:
      - 'one_step': x,y are one-step pairs. x: (B, N_points, 1), y: (B, N_points, 1)
      - 'rollout': train/valid use one-step; test may use rollout targets:
           x: (B, N_points, 1), y: (B, S, N_points, 1)

    Geometry:
      - self.geom: {'dim': 3, 'layout': 'grid', 'spatial_shape': (X',Y',Z'),
                    'sample_factor': (sx,sy,sz), 'axis_order': 'xyz', 'prop': prop}
      - self.coords: (X'*Y'*Z', 3)
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        self.sample_factor: Tuple[int, int, int] = _as_3tuple(data_args.get("sample_factor", (1, 1, 1)))

        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))

        self.prop: str = str(data_args.get("prop", "Vx"))

        self.in_t: int = int(data_args.get("in_t", 0))
        self.out_t: int = int(data_args.get("out_t", 1))
        self.duration: int = int(data_args.get("duration", 1))

        self.task: str = str(data_args.get("task", "one_step"))  # "one_step" | "rollout"
        self.rollout_steps: int = int(data_args.get("rollout_steps", self.duration))
        self.rollout_stride: int = int(data_args.get("rollout_stride", 0))

        # Coord handling
        self.use_file_coords: bool = bool(data_args.get("use_file_coords", True))
        self.coord_normalize: bool = bool(data_args.get("coord_normalize", True))

        # 1D coordinate vectors (filled in load_raw_data if available)
        self._x1d: Optional[torch.Tensor] = None
        self._y1d: Optional[torch.Tensor] = None
        self._z1d: Optional[torch.Tensor] = None
        self._t1d: Optional[torch.Tensor] = None

        super().__init__(data_args, **kwargs)

    def get_cache_path(self) -> str:
        if not self.data_path:
            return "ns3d_processed.pt"

        root, _ = osp.splitext(self.data_path)
        sx, sy, sz = self.sample_factor
        sf_tag = f"{sx}-{sy}-{sz}"
        norm_tag = self.normalizer_type if self.normalize else "none"

        task_tag = self.task
        if task_tag == "rollout":
            extra = f"_ro{self.rollout_steps}_rs{self.rollout_stride}"
        else:
            extra = ""

        return (
            f"{root}_{self.prop}"
            f"_sf{sf_tag}"
            f"_it{self.in_t}_ot{self.out_t}_dur{self.duration}"
            f"{extra}_task{task_tag}_norm{norm_tag}_processed.pt"
        )

    def load_raw_data(self, **kwargs: Any) -> torch.Tensor:
        data_path = self.data_path
        if not data_path:
            raise ValueError("data_path is empty.")

        # HDF5
        with File(data_path, "r") as f:
            if self.prop not in f:
                raise KeyError(
                    f"Key '{self.prop}' not found in HDF5 file. Available keys: {list(f.keys())}"
                )
            node = f.get(self.prop, None)
            if node is None or not isinstance(node, H5Dataset):
                raise TypeError(f"Expected dataset '{self.prop}' in HDF5 file.")
            var = np.asarray(node[()])

            # Load 1D coords if present
            if self.use_file_coords:
                xk = _first_existing_key(f, ["x-coordinate", "x_coordinate", "x", "X"])
                yk = _first_existing_key(f, ["y-coordinate", "y_coordinate", "y", "Y"])
                zk = _first_existing_key(f, ["z-coordinate", "z_coordinate", "z", "Z"])
                tk = _first_existing_key(f, ["t-coordinate", "t_coordinate", "t", "T", "time"])

                def _load_1d(k: Optional[str]) -> Optional[torch.Tensor]:
                    if k is None:
                        return None
                    arr1d = np.asarray(f[k][()])
                    arr1d = np.ravel(arr1d)
                    return torch.tensor(arr1d, dtype=torch.float32)

                self._x1d = _load_1d(xk)
                self._y1d = _load_1d(yk)
                self._z1d = _load_1d(zk)
                self._t1d = _load_1d(tk)

        if var.ndim != 5:
            raise ValueError(f"Expected 5D array for '{self.prop}', got shape={var.shape}.")

        x_len = int(self._x1d.numel()) if self._x1d is not None else None
        y_len = int(self._y1d.numel()) if self._y1d is not None else None
        z_len = int(self._z1d.numel()) if self._z1d is not None else None
        t_len = int(self._t1d.numel()) if self._t1d is not None else None

        var = _standardize_to_n_t_xyz(var, x_len=x_len, y_len=y_len, z_len=z_len, t_len=t_len)
        return torch.tensor(var, dtype=torch.float32)  # (N,T,X,Y,Z)

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
            data_split: (N, T, X, Y, Z)

        Returns:
            train/valid (always one_step):
                x: (B, N_points, 1), y: (B, N_points, 1)
            test, task="one_step":
                x: (B, N_points, 1), y: (B, N_points, 1)
            test, task="rollout":
                x: (B, N_points, 1), y: (B, S, N_points, 1)
        """
        if data_split.ndim != 5:
            raise ValueError(f"Expected (N,T,X,Y,Z), got {tuple(data_split.shape)}.")
        if self.out_t <= 0:
            raise ValueError(f"out_t must be > 0, got {self.out_t}.")

        in_t = int(self.in_t)
        out_t = int(self.out_t)
        duration = int(self.duration)

        if in_t < 0:
            raise ValueError(f"in_t must be >= 0, got {in_t}.")
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}.")

        # train/valid always one-step; rollout only used for test (match NS2D)
        effective_task = "one_step" if mode in ("train", "valid") else self.task

        N, T, X, Y, Z = data_split.shape

        if effective_task == "one_step":
            end_t_task = in_t + duration + out_t
        else:
            steps = int(self.rollout_steps)
            if steps <= 0:
                raise ValueError(f"rollout_steps must be > 0, got {steps}.")
            end_t_task = in_t + steps * out_t + 1

        if end_t_task > T:
            raise ValueError(f"Temporal window exceeds T: end_t_task={end_t_task} > T={T}.")

        end_t_norm = in_t + duration + out_t
        if end_t_norm > T:
            end_t_norm = end_t_task

        data_norm = data_split[:, in_t:end_t_norm, :, :, :]  # (N,Tn,X,Y,Z)
        data_task = data_split[:, in_t:end_t_task, :, :, :]  # (N,Tt,X,Y,Z)

        # spatial subsampling
        sx, sy, sz = self.sample_factor
        if sx > 1 or sy > 1 or sz > 1:
            data_norm = data_norm[:, :, ::sx, ::sy, ::sz]
            data_task = data_task[:, :, ::sx, ::sy, ::sz]

        n, t_norm, x_, y_, z_ = data_norm.shape
        _, t_task, _, _, _ = data_task.shape

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
            data_task = x_normalizer.encode(flat_task).reshape(n, t_task, x_, y_, z_)

        # --- geometry / coords (once) ---
        if self.geom is None or self.coords is None:
            self.geom = {
                "dim": 3,
                "layout": "grid",
                "spatial_shape": (x_, y_, z_),
                "sample_factor": (sx, sy, sz),
                "axis_order": "xyz",
                "prop": self.prop,
            }

            eps = 1e-12

            def _norm_1d(v: torch.Tensor) -> torch.Tensor:
                if not self.coord_normalize:
                    return v
                vmin = float(v.min().item())
                vmax = float(v.max().item())
                if abs(vmax - vmin) < eps:
                    return torch.zeros_like(v)
                return (v - vmin) / (vmax - vmin)

            if self.use_file_coords and (self._x1d is not None and self._y1d is not None and self._z1d is not None):
                xv = _norm_1d(self._x1d[::sx].clone())
                yv = _norm_1d(self._y1d[::sy].clone())
                zv = _norm_1d(self._z1d[::sz].clone())
            else:
                xv = torch.linspace(0.0, 1.0, x_, dtype=torch.float32)
                yv = torch.linspace(0.0, 1.0, y_, dtype=torch.float32)
                zv = torch.linspace(0.0, 1.0, z_, dtype=torch.float32)

            grid_x, grid_y, grid_z = torch.meshgrid(xv, yv, zv, indexing="ij")
            self.coords = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)

        n_points = x_ * y_ * z_

        # --- one-step branch ---
        if effective_task == "one_step":
            if t_task < duration + out_t:
                raise ValueError(
                    f"Not enough temporal length for one_step: "
                    f"t_task={t_task}, duration={duration}, out_t={out_t}."
                )

            x_u = data_task[:, :duration, :, :, :]              # (N,duration,X',Y',Z')
            y_u = data_task[:, out_t:out_t + duration, :, :, :] # (N,duration,X',Y',Z')

            x_u = x_u.flatten(0, 1).contiguous()  # (B,X',Y',Z')
            y_u = y_u.flatten(0, 1).contiguous()  # (B,X',Y',Z')

            B = int(x_u.shape[0])

            x = x_u.view(B, n_points, 1)
            y = y_u.view(B, n_points, 1)
            return x, y, x_normalizer, y_normalizer

        # --- rollout branch (test only) ---
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
            u0 = data_task[:, s, :, :, :]  # (N,X',Y',Z')
            gt = data_task[:, s + out_t : s + (steps + 1) * out_t : out_t, :, :, :]  # (N,steps,X',Y',Z')
            xs.append(u0)
            ys.append(gt)

        x_seq = torch.cat(xs, dim=0).contiguous()  # (B',X',Y',Z')
        y_seq = torch.cat(ys, dim=0).contiguous()  # (B',steps,X',Y',Z')

        B_total = int(x_seq.shape[0])

        x = x_seq.view(B_total, n_points, 1)
        y = y_seq.view(B_total, steps, n_points, 1)
        return x, y, x_normalizer, y_normalizer
