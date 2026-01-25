# datasets/carra.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence
import os.path as osp

import torch

from .base import BaseDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


def _as_2tuple(sample_factor: Any) -> Tuple[int, int]:
    """Accept int / list / tuple and return (sx, sy)."""
    if sample_factor is None:
        return (1, 1)
    if isinstance(sample_factor, int):
        sf = int(sample_factor)
        return (sf, sf)
    if isinstance(sample_factor, (list, tuple)):
        if len(sample_factor) != 2:
            raise ValueError(f"sample_factor must have length 2, got {sample_factor}.")
        return (int(sample_factor[0]), int(sample_factor[1]))
    raise TypeError(f"Unsupported type for sample_factor: {type(sample_factor)}")


class CarraDataset(BaseDataset):
    """
    CARRA dataset (time-series on lat-lon grid).

    Raw data file (torch.save) is expected to contain:
      data, lat, lon = torch.load(data_path)
    where:
      - data: Tensor with shape (2, T, H, W) or (2, H, W, T) (we only support common (2,T,H,W) here)
      - lat, lon: Tensor with shape (H, W)

    prop:
      - "v10" -> data[0]
      - "sp"  -> data[1]

    Output (processed) is always BNC:
      x: (B, N_points, Cx)
      y: (B, N_points, 1)

    coords / geom:
      - self.coords: (N_points, 2) from (lat, lon) after subsampling + normalization
      - self.geom: {"dim":2, "layout":"grid", "spatial_shape":(H',W'), "sample_factor":(sx,sy), "prop":prop}

    coords_in_x:
      - False (default): x has Cx=1, coords provided via self.coords (trainer passes coords to model).
      - True: x has Cx=3 by concatenating [lat, lon, value] (legacy behavior).
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        # Backward-compatible "sub" (float) -> BaseDataset subset
        sub = data_args.get("sub", False)
        if sub is not False and sub is not None:
            data_args["subset"] = True
            data_args["subset_ratio"] = float(sub)

        self.sample_factor: Tuple[int, int] = _as_2tuple(data_args.get("sample_factor", (1, 1)))
        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))

        self.prop: str = str(data_args.get("prop", "v10"))  # v10 | sp

        # temporal pairing controls (generalization of x=data[:-1], y=data[1:])
        self.in_t: int = int(data_args.get("in_t", 0))
        self.out_t: int = int(data_args.get("out_t", 1))
        # duration<=0 means "use all available pairs in this split"
        self.duration: int = int(data_args.get("duration", -1))

        # coordinate handling
        self.coords_in_x: bool = bool(data_args.get("coords_in_x", False))
        self.coord_normalize: bool = bool(data_args.get("coord_normalize", True))
        # "zscore" matches你的旧实现；也支持 "minmax"
        self.coord_norm_type: str = str(data_args.get("coord_norm_type", "zscore"))

        # filled in load_raw_data()
        self._lat2d: Optional[torch.Tensor] = None
        self._lon2d: Optional[torch.Tensor] = None

        super().__init__(data_args, **kwargs)

    def get_cache_path(self) -> str:
        if not self.data_path:
            return "carra_processed.pt"

        root, _ = osp.splitext(self.data_path)
        sx, sy = self.sample_factor
        sf_tag = f"{sx}-{sy}"
        norm_tag = self.normalizer_type if self.normalize else "none"
        coord_tag = f"{self.coord_norm_type}" if self.coord_normalize else "coord_none"
        cinx = "cinx1" if self.coords_in_x else "cinx0"

        return (
            f"{root}_{self.prop}"
            f"_sf{sf_tag}"
            f"_it{self.in_t}_ot{self.out_t}_dur{self.duration}"
            f"_{cinx}_{coord_tag}_norm{norm_tag}_processed.pt"
        )

    def load_raw_data(self, **kwargs: Any) -> torch.Tensor:
        """
        Return a tensor with shape (T, H, W) for the selected prop.
        Also stores lat/lon (H, W) in self._lat2d / self._lon2d.
        """
        if not self.data_path:
            raise ValueError("data_path is empty.")

        obj = torch.load(self.data_path)
        if not isinstance(obj, (tuple, list)) or len(obj) != 3:
            raise ValueError("Expected torch.load(data_path) -> (data, lat, lon).")

        data, lat, lon = obj
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor.")
        if not isinstance(lat, torch.Tensor) or not isinstance(lon, torch.Tensor):
            raise TypeError("lat/lon must be torch.Tensor.")

        # lat/lon should be (H, W)
        if lat.ndim != 2 or lon.ndim != 2:
            raise ValueError(f"Expected lat/lon to be 2D (H,W), got lat={lat.shape}, lon={lon.shape}.")

        self._lat2d = lat.float()
        self._lon2d = lon.float()

        # data should have a channel dimension for variables
        if data.ndim != 4:
            raise ValueError(f"Expected data to be 4D, got shape={tuple(data.shape)}.")

        # common layout: (2, T, H, W)
        if data.shape[0] != 2:
            raise ValueError(f"Expected data.shape[0]==2 (v10, sp), got {data.shape[0]}.")

        if self.prop == "v10":
            field = data[0]
        elif self.prop == "sp":
            field = data[1]
        else:
            raise ValueError("Invalid prop. Choose from 'v10' or 'sp'.")

        # Now field should be (T,H,W)
        if field.ndim != 3:
            raise ValueError(f"Expected selected field to be 3D (T,H,W), got {tuple(field.shape)}.")

        return field.float()

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N,2)
        """
        if not self.coord_normalize:
            return coords

        if self.coord_norm_type == "zscore":
            mean = coords.mean(dim=0, keepdim=True)
            std = coords.std(dim=0, keepdim=True)
            std = torch.where(std > 1e-12, std, torch.ones_like(std))
            return (coords - mean) / std

        if self.coord_norm_type == "minmax":
            cmin = coords.min(dim=0, keepdim=True).values
            cmax = coords.max(dim=0, keepdim=True).values
            denom = torch.where((cmax - cmin) > 1e-12, (cmax - cmin), torch.ones_like(cmax))
            return (coords - cmin) / denom

        raise ValueError(f"Unknown coord_norm_type: {self.coord_norm_type} (use 'zscore' or 'minmax').")

    def process_split(
        self,
        data_split: torch.Tensor,
        mode: str,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any], Optional[Any]]:
        """
        data_split: (T_split, H, W)
        returns:
          x: (B, N_points, Cx)
          y: (B, N_points, 1)
        """
        if data_split.ndim != 3:
            raise ValueError(f"Expected (T,H,W), got {tuple(data_split.shape)}.")
        if self.out_t <= 0:
            raise ValueError(f"out_t must be > 0, got {self.out_t}.")

        T, H, W = data_split.shape
        in_t = int(self.in_t)
        out_t = int(self.out_t)

        if in_t < 0:
            raise ValueError(f"in_t must be >= 0, got {in_t}.")
        if in_t >= T:
            raise ValueError(f"in_t={in_t} must be < T={T}.")

        # pair range: [t0, t1)
        t0 = in_t
        t1 = T - out_t
        if t1 <= t0:
            raise ValueError(f"Not enough time steps: T={T}, in_t={in_t}, out_t={out_t}.")

        if self.duration > 0:
            t1 = min(t1, t0 + int(self.duration))

        x_u = data_split[t0:t1]               # (B,H,W)
        y_u = data_split[t0 + out_t : t1 + out_t]  # (B,H,W)

        # spatial subsample
        sx, sy = self.sample_factor
        if sx > 1 or sy > 1:
            x_u = x_u[:, ::sx, ::sy]
            y_u = y_u[:, ::sx, ::sy]

        B, h, w = x_u.shape
        n_points = h * w

        # build coords/geom once (after subsampling so shapes match)
        if self.geom is None or self.coords is None:
            if self._lat2d is None or self._lon2d is None:
                raise RuntimeError("lat/lon not loaded.")

            lat = self._lat2d[::sx, ::sy].contiguous()
            lon = self._lon2d[::sx, ::sy].contiguous()
            if lat.shape != (h, w) or lon.shape != (h, w):
                raise ValueError(f"lat/lon shape mismatch after subsample: lat={lat.shape}, expected {(h,w)}.")

            coords = torch.stack((lat, lon), dim=-1).view(-1, 2)  # (N,2)
            coords = self._normalize_coords(coords)

            self.coords = coords
            self.geom = {
                "dim": 2,
                "layout": "grid",
                "spatial_shape": (h, w),
                "sample_factor": (sx, sy),
                "prop": self.prop,
            }

        # to BNC (value channel only)
        x = x_u.unsqueeze(-1).contiguous().view(B, n_points, 1)  # (B,N,1)
        y = y_u.unsqueeze(-1).contiguous().view(B, n_points, 1)  # (B,N,1)

        # normalization on value channels
        if self.normalize:
            if mode == "train":
                if self.normalizer_type == "PGN":
                    x_normalizer = UnitGaussianNormalizer(x)
                    y_normalizer = UnitGaussianNormalizer(y)
                else:
                    x_normalizer = GaussianNormalizer(x)
                    y_normalizer = GaussianNormalizer(y)
            else:
                if x_normalizer is None or y_normalizer is None:
                    raise RuntimeError("Normalizer is None for non-train split.")

            x = x_normalizer.encode(x)
            y = y_normalizer.encode(y)

        # optional legacy behavior: concat coords into x channels -> (B,N,3)
        if self.coords_in_x:
            assert self.coords is not None
            coords_b = self.coords.unsqueeze(0).repeat(B, 1, 1)  # (B,N,2)
            x = torch.cat([coords_b, x], dim=-1)

        return x, y, x_normalizer, y_normalizer
