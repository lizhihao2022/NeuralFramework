# datasets/airfoil_time.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import os.path as osp
import numpy as np
import torch

from .base import BaseDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


def _to_tn1(x: torch.Tensor) -> torch.Tensor:
    """
    Force (T, N, 1).
    Accepts:
      - (T, N)   -> (T, N, 1)
      - (T, N, 1)-> (T, N, 1)
    """
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim == 3 and x.shape[-1] == 1:
        return x
    raise ValueError(f"Expected (T,N) or (T,N,1), got shape={tuple(x.shape)}")


def _normalize_coords(coords: torch.Tensor, norm_type: str = "zscore") -> torch.Tensor:
    """
    coords: (N, d)
    """
    eps = 1e-12
    if norm_type == "none":
        return coords
    if norm_type == "zscore":
        mean = coords.mean(dim=0, keepdim=True)
        std = coords.std(dim=0, keepdim=True)
        std = torch.where(std > eps, std, torch.ones_like(std))
        return (coords - mean) / std
    if norm_type == "minmax":
        cmin = coords.min(dim=0, keepdim=True).values
        cmax = coords.max(dim=0, keepdim=True).values
        denom = torch.where((cmax - cmin) > eps, (cmax - cmin), torch.ones_like(cmax))
        return (coords - cmin) / denom
    raise ValueError(f"Unknown coord_norm_type: {norm_type} (use 'zscore'|'minmax'|'none').")


def _subsample_nodes(
    coords: torch.Tensor,
    u: torch.Tensor,
    cell: Optional[torch.Tensor],
    sample_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    coords: (N, d)
    u:      (T, N, 1)
    cell:   (F, k) or None
    """
    if sample_factor <= 1:
        return coords, u, cell

    N = coords.shape[0]
    keep = torch.arange(0, N, sample_factor, device=coords.device)
    coords2 = coords.index_select(0, keep)
    u2 = u.index_select(1, keep)

    if cell is None:
        return coords2, u2, None

    # remap cell indices: old -> new
    mapping = torch.full((N,), -1, dtype=torch.long, device=coords.device)
    mapping[keep] = torch.arange(keep.numel(), device=coords.device)

    cell_mapped = mapping[cell]            # (F, k)
    valid = (cell_mapped >= 0).all(dim=1)  # drop faces that reference removed nodes
    cell2 = cell_mapped[valid].contiguous()

    return coords2, u2, cell2


class AirfoilTimeDataset(BaseDataset):
    """
    Airfoil time dataset (NO torch_geometric), NS2D-style task behavior.

    Raw directory layout:
      data_path/
        train_0.npy ... train_4.npy (optional)
        valid.npy
        test.npy

    Each entry is a dict-like object with keys:
      - 'density':   (T, N) or (T, N, 1)
      - 'pressure':  (T, N) or (T, N, 1)
      - 'velocity':  (T, N, Cv)   (Cv>=2)
      - 'mesh_pos':  (1, N, d) or (N, d)
      - 'cells':     (1, F, k) or (F, k) (optional)

    Hyper-params:
      - prop: 'density'|'pressure'|'velocity'
      - velocity_component: required if prop=='velocity', in {0,1,2}
      - task: 'one_step'|'rollout' (like NS2D; only affects test)
      - in_t: start time index (like NS2D)
      - out_t: step size
      - duration: number of one-step pairs used to form training samples
      - rollout_steps / rollout_stride: for test rollout

    Outputs:
      one_step: x (B,N,1), y (B,N,1)
      rollout : x (B,N,1), y (B,S,N,1)
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        # --- task controls (NS2D-style) ---
        self.task: str = str(data_args.get("task", "one_step"))
        if self.task not in ("one_step", "rollout"):
            raise ValueError("task must be 'one_step' or 'rollout'.")

        self.rollout_steps: int = int(data_args.get("rollout_steps", 20))
        self.rollout_stride: int = int(data_args.get("rollout_stride", 0))

        # --- temporal window controls ---
        self.in_t: int = int(data_args.get("in_t", 0))            # start index
        self.out_t: int = int(data_args.get("out_t", 1))          # step
        self.duration: int = int(data_args.get("duration", data_args.get("duration_t", 10)))

        # --- spatial subsampling ---
        self.sample_factor: int = int(data_args.get("sample_factor", 1))

        # --- normalization ---
        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))

        # --- single-variable task (STRICT y: 1 channel) ---
        self.prop: str = str(data_args.get("prop", "density"))
        if self.prop not in ("density", "pressure", "velocity"):
            raise ValueError("prop must be one of {'density','pressure','velocity'}.")

        self.velocity_component: Optional[int] = data_args.get("velocity_component", None)
        if self.prop == "velocity":
            if self.velocity_component is None:
                raise ValueError(
                    "Strict single-channel requirement: for prop='velocity', "
                    "you MUST set velocity_component (e.g., 0/1/2)."
                )
            self.velocity_component = int(self.velocity_component)

        # --- coords normalization (optional) ---
        self.coord_normalize: bool = bool(data_args.get("coord_normalize", True))
        self.coord_norm_type: str = str(data_args.get("coord_norm_type", "zscore"))

        # internal ref for mesh consistency check
        self._coords_ref: Optional[torch.Tensor] = None
        self._cell_ref: Optional[torch.Tensor] = None

        super().__init__(data_args, **kwargs)

    def get_cache_path(self) -> str:
        # data_path is a directory
        root = self.data_path or "."
        sf = self.sample_factor
        prop_tag = self.prop if self.prop != "velocity" else f"velocity{int(self.velocity_component)}"
        norm_tag = self.normalizer_type if self.normalize else "none"
        return osp.join(
            root,
            f"airfoil_{prop_tag}_sf{sf}_it{self.in_t}_ot{self.out_t}_dur{self.duration}_"
            f"task{self.task}_steps{self.rollout_steps}_stride{self.rollout_stride}_"
            f"coord{int(self.coord_normalize)}_{self.coord_norm_type}_norm{norm_tag}_processed.pt",
        )

    def load_raw_data(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.data_path:
            raise ValueError("data_path is empty.")
        if not osp.isdir(self.data_path):
            raise ValueError(f"AirfoilTimeDataset expects a directory, got: {self.data_path}")

        # train shards
        train_parts = []
        p0 = osp.join(self.data_path, "train_0.npy")
        if not osp.exists(p0):
            raise FileNotFoundError(f"Missing {p0}")
        train_parts.append(np.load(p0, allow_pickle=True))
        for i in range(1, 5):
            pi = osp.join(self.data_path, f"train_{i}.npy")
            if osp.exists(pi):
                train_parts.append(np.load(pi, allow_pickle=True))
        train = np.concatenate(train_parts, axis=0)

        valid_path = osp.join(self.data_path, "valid.npy")
        test_path = osp.join(self.data_path, "test.npy")
        if not osp.exists(valid_path):
            raise FileNotFoundError(f"Missing {valid_path}")
        if not osp.exists(test_path):
            raise FileNotFoundError(f"Missing {test_path}")

        valid = np.load(valid_path, allow_pickle=True)
        test = np.load(test_path, allow_pickle=True)

        return {"train": train, "valid": valid, "test": test}

    def split_data(self, raw: Any) -> Tuple[Any, Any, Any]:
        # raw already contains explicit splits
        if not isinstance(raw, dict) or not all(k in raw for k in ("train", "valid", "test")):
            raise TypeError("AirfoilTimeDataset.load_raw_data must return dict with keys {'train','valid','test'}.")
        return raw["train"], raw["valid"], raw["test"]

    def _extract_u_pos_cell(self, sample: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # mesh_pos
        pos = torch.from_numpy(sample["mesh_pos"]).float()
        if pos.ndim == 3:
            pos = pos[0]
        if pos.ndim != 2:
            raise ValueError(f"mesh_pos should be (N,d) or (1,N,d), got {tuple(pos.shape)}")

        # optional cells
        cell = None
        if "cells" in sample:
            c = torch.from_numpy(sample["cells"])
            if c.ndim == 3:
                c = c[0]
            if c.ndim == 2:
                cell = c.long()

        # pick variable -> u: (T,N,1)
        if self.prop == "density":
            u = _to_tn1(torch.from_numpy(sample["density"]).float())
        elif self.prop == "pressure":
            u = _to_tn1(torch.from_numpy(sample["pressure"]).float())
        else:  # velocity
            v = torch.from_numpy(sample["velocity"]).float()
            if v.ndim == 2:
                # (T,N) -> (T,N,1) but velocity should not be 2D; still allow
                u = v.unsqueeze(-1)
            elif v.ndim == 3:
                k = int(self.velocity_component)  # guaranteed not None
                if k < 0 or k >= v.shape[-1]:
                    raise ValueError(f"velocity_component={k} out of range for velocity dim={v.shape[-1]}")
                u = v[..., k:k + 1]  # (T,N,1)
            else:
                raise ValueError(f"velocity expected (T,N,C), got {tuple(v.shape)}")

            # enforce single-channel
            if u.shape[-1] != 1:
                raise RuntimeError("Internal error: velocity slicing did not produce single-channel output.")

        return u, pos, cell

    def process_split(
        self,
        data_split: Any,
        mode: str,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any], Optional[Any]]:
        """
        Returns:
            train (any task): x (B,N,1), y (B,N,1)
            valid/test, task="one_step": x (B,N,1), y (B,N,1)
            test, task="rollout": x (B,N,1), y (B,S,N,1)
        """
        if self.out_t <= 0:
            raise ValueError(f"out_t must be > 0, got {self.out_t}.")
        if self.in_t < 0:
            raise ValueError(f"in_t must be >= 0, got {self.in_t}.")
        if self.duration <= 0:
            raise ValueError(f"duration must be > 0, got {self.duration}.")

        # train/valid always use one_step; rollout only used for test
        effective_task = "one_step" if mode in ("train", "valid") else self.task

        # collect all trajectories into a tensor (Ntraj, T, N, 1) and ensure fixed mesh
        traj_u: List[torch.Tensor] = []
        coords_ref: Optional[torch.Tensor] = None
        cell_ref: Optional[torch.Tensor] = None

        for i in range(len(data_split)):
            sample = data_split[i]
            u, pos, cell = self._extract_u_pos_cell(sample)

            # subsample nodes
            pos, u, cell = _subsample_nodes(pos, u, cell, self.sample_factor)

            # coords normalization (consistent across dataset)
            if self.coord_normalize:
                pos = _normalize_coords(pos, self.coord_norm_type)

            if coords_ref is None:
                coords_ref = pos
                cell_ref = cell
            else:
                # strict consistency check for global coords injection
                if pos.shape != coords_ref.shape:
                    raise ValueError(
                        "Airfoil mesh_pos is not consistent across samples (shape mismatch). "
                        "Current framework expects fixed coords for rollout."
                    )
                max_err = (pos - coords_ref).abs().max().item()
                if max_err > 1e-5:
                    raise ValueError(
                        f"Airfoil mesh_pos varies across samples (max |Δ|={max_err:.2e}). "
                        "Current rollout pipeline requires a fixed mesh. "
                        "Either pre-align to a common mesh or redesign rollout to carry per-sample coords."
                    )

            traj_u.append(u)

        u_all = torch.stack(traj_u, dim=0)  # (Ntraj, T, N, 1)
        n_traj, T, n_points, _ = u_all.shape

        # decide temporal window lengths (follow NS2D)
        if effective_task == "one_step":
            end_t_task = self.in_t + self.duration + self.out_t
        else:
            steps = int(self.rollout_steps)
            if steps <= 0:
                raise ValueError(f"rollout_steps must be > 0, got {steps}.")
            end_t_task = self.in_t + steps * self.out_t + 1

        if end_t_task > T:
            raise ValueError(f"Temporal window exceeds T: end_t_task={end_t_task} > T={T}.")

        end_t_norm = self.in_t + self.duration + self.out_t
        if end_t_norm > T:
            end_t_norm = end_t_task

        # windows: (Ntraj, t, N)
        data_norm = u_all[:, self.in_t:end_t_norm, :, 0]  # (Ntraj, t_norm, N)
        data_task = u_all[:, self.in_t:end_t_task, :, 0]  # (Ntraj, t_task, N)

        t_norm = int(data_norm.shape[1])
        t_task = int(data_task.shape[1])

        # --- normalization (like NS2D): normalizer on data_norm, apply to data_task ---
        if self.normalize:
            flat_norm = data_norm.reshape(n_traj * t_norm, n_points).unsqueeze(-1)  # (Ntraj*t_norm, N, 1)

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

            flat_task = data_task.reshape(n_traj * t_task, n_points).unsqueeze(-1)  # (Ntraj*t_task, N, 1)
            flat_task = x_normalizer.encode(flat_task)                               # (Ntraj*t_task, N, 1)
            data_task = flat_task.view(n_traj, t_task, n_points)                     # (Ntraj, t_task, N)

        # --- geometry / coords (once) ---
        if self.geom is None or self.coords is None:
            if coords_ref is None:
                raise RuntimeError("coords_ref is None unexpectedly.")

            self.coords = coords_ref  # (N, d)
            self.geom = {
                "dim": int(coords_ref.shape[-1]),
                "layout": "mesh",
                "n_points": int(n_points),
                "sample_factor": int(self.sample_factor),
                "prop": self.prop if self.prop != "velocity" else f"velocity{int(self.velocity_component)}",
            }
            if cell_ref is not None:
                self.geom["cell"] = cell_ref

        # --- one-step (train/valid always; test if task=one_step) ---
        if effective_task == "one_step":
            if t_task < self.duration + self.out_t:
                raise ValueError(
                    f"Not enough temporal length for one_step: "
                    f"t_task={t_task}, duration={self.duration}, out_t={self.out_t}."
                )

            x_u = data_task[:, : self.duration, :]                         # (Ntraj, duration, N)
            y_u = data_task[:, self.out_t : self.out_t + self.duration, :] # (Ntraj, duration, N)

            x_u = x_u.flatten(0, 1)  # (B, N)
            y_u = y_u.flatten(0, 1)  # (B, N)

            B = int(x_u.shape[0])
            x = x_u.view(B, n_points, 1)
            y = y_u.view(B, n_points, 1)
            return x, y, x_normalizer, y_normalizer

        # --- rollout (ONLY test when task=rollout) ---
        steps = int(self.rollout_steps)
        max_start = (t_task - 1) - steps * self.out_t
        if max_start < 0:
            raise ValueError(
                f"Not enough temporal length for rollout: "
                f"t_task={t_task}, steps={steps}, out_t={self.out_t}."
            )

        if self.rollout_stride and self.rollout_stride > 0:
            starts = list(range(0, max_start + 1, self.rollout_stride))
        else:
            starts = [0]

        xs = []
        ys = []
        for s in starts:
            u0 = data_task[:, s, :]  # (Ntraj, N)
            gt = data_task[
                :,
                s + self.out_t : s + (steps + 1) * self.out_t : self.out_t,
                :,
            ]  # (Ntraj, steps, N)

            xs.append(u0)
            ys.append(gt)

        x_seq = torch.cat(xs, dim=0)  # (B', N)
        y_seq = torch.cat(ys, dim=0)  # (B', steps, N)

        B_total = int(x_seq.shape[0])
        x = x_seq.view(B_total, n_points, 1)
        y = y_seq.view(B_total, steps, n_points, 1)
        return x, y, x_normalizer, y_normalizer
