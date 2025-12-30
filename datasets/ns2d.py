# datasets/ns_2d.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import scipy.io as sio
from h5py import File, Dataset as H5Dataset

from .base import BaseDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class NS2DDataset(BaseDataset):
    """
    2D Navier–Stokes dataset.

    Raw file is expected to contain key 'u':
      - For .mat: raw_data['u'] of shape (N, H, W, T)
      - For .h5:  commonly one of:
          (H, W, T, N)  or  (T, H, W, N)  or  (N, H, W, T)

    After loading, we standardize raw tensor to shape (N, H, W, T).
    """

    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        # NS-specific options
        self.sample_factor: int = int(data_args.get("sample_factor", 1))
        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))

        # temporal slicing options (new feature)
        self.in_t: int = int(data_args.get("in_t", 5))
        self.out_t: int = int(data_args.get("out_t", 1))
        self.duration: int = int(data_args.get("duration", 10))

        super().__init__(data_args, **kwargs)

    def get_cache_path(self) -> str:
        # encode preprocessing knobs into cache name
        if not self.data_path:
            return "ns2d_processed.pt"

        root, _ = __import__("os.path").path.splitext(self.data_path)
        sf = self.sample_factor
        norm_tag = self.normalizer_type if self.normalize else "none"
        return f"{root}_sf{sf}_it{self.in_t}_ot{self.out_t}_dur{self.duration}_norm{norm_tag}_processed.pt"

    # ------------ override hooks from BaseDataset ------------
    def load_raw_data(self, **kwargs: Any) -> torch.Tensor:
        """
        Load raw Navier–Stokes data as a torch.Tensor of shape (N, H, W, T).
        Supports .mat and .h5.
        """
        data_path = self.data_path
        if not data_path:
            raise ValueError("data_path is empty.")

        # --- try MAT first ---
        try:
            raw_data = sio.loadmat(data_path)
            if "u" not in raw_data:
                raise KeyError("Key 'u' not found in .mat file.")
            data = torch.tensor(raw_data["u"], dtype=torch.float32)  # (N,H,W,T)
            if data.ndim != 4:
                raise ValueError(f"Expected 'u' to be 4D in .mat, got shape={tuple(data.shape)}.")
            return data
        except Exception:
            pass

        # --- HDF5 fallback ---
        with File(data_path, "r") as raw_data:
            u_node = raw_data.get("u", None)
            if u_node is None:
                raise KeyError("Key 'u' not found in HDF5 file.")
            if not isinstance(u_node, H5Dataset):
                raise TypeError("Expected dataset 'u' in HDF5 file.")
            u_array = np.asarray(u_node[()])

        if u_array.ndim != 4:
            raise ValueError(f"Expected 'u' to be 4D in HDF5, got shape={u_array.shape}.")

        # Heuristic standardization to (N,H,W,T)
        # Typical NS: H,W ~ 32/64/128, T ~ 10~50, N ~ hundreds/thousands
        d0, d1, d2, d3 = u_array.shape

        def _is_hw(x: int) -> bool:
            return x in (16, 32, 48, 64, 96, 128, 256) or x >= 16

        def _is_t(x: int) -> bool:
            return 1 <= x <= 256

        def _is_n(x: int) -> bool:
            return x >= 64

        if _is_hw(d0) and _is_hw(d1) and _is_t(d2) and _is_n(d3):
            # (H, W, T, N) -> (N, H, W, T)
            std = np.transpose(u_array, (3, 0, 1, 2))
        elif _is_t(d0) and _is_hw(d1) and _is_hw(d2) and _is_n(d3):
            # (T, H, W, N) -> (N, H, W, T)
            std = np.transpose(u_array, (3, 1, 2, 0))
        elif _is_n(d0) and _is_hw(d1) and _is_hw(d2) and _is_t(d3):
            # already (N, H, W, T)
            std = u_array
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
        x_normalizer: Optional[torch.nn.Module] = None,
        y_normalizer: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """
        Pre-process a split block.

        Args:
            data_split: (N, H, W, T)
            mode: 'train' | 'valid' | 'test'

        Returns:
            x: (B, H', W', C_in=3)  where channels are [grid_x, grid_y, u]
            y: (B, H', W', C_out=1)
            x_normalizer, y_normalizer
        """
        if data_split.ndim != 4:
            raise ValueError(f"Expected data_split shape (N,H,W,T), got {tuple(data_split.shape)}.")

        in_t = self.in_t
        out_t = self.out_t
        duration = self.duration

        if in_t < 0 or out_t <= 0 or duration <= 0:
            raise ValueError(f"Invalid (in_t,out_t,duration)=({in_t},{out_t},{duration}).")

        # (N, H, W, T) -> select window -> (N, H, W, T')
        end_t = in_t + duration + out_t
        if end_t > data_split.shape[-1]:
            raise ValueError(
                f"Temporal window exceeds T: in_t+duration+out_t={end_t} > T={data_split.shape[-1]}."
            )
        data = data_split[..., in_t:end_t]

        # (N, H, W, T') -> (N, T', H, W)
        data = data.permute(0, 3, 1, 2)

        # optional spatial subsampling
        sf = self.sample_factor
        if sf > 1:
            data = data[:, :, ::sf, ::sf]

        n, t_sel, h, w = data.shape

        # normalize on (N*T, P, 1)
        if self.normalize:
            flat = data.reshape(n * t_sel, -1, 1)
            if mode == "train":
                if self.normalizer_type == "PGN":
                    x_normalizer = UnitGaussianNormalizer(flat)
                    y_normalizer = x_normalizer
                else:
                    x_normalizer = GaussianNormalizer(flat)
                    y_normalizer = x_normalizer
            else:
                if x_normalizer is None or y_normalizer is None:
                    raise RuntimeError("Normalizer is None in non-train mode with normalize=True.")
            data = x_normalizer.encode(flat).reshape(n, t_sel, h, w)

        # build (x,y) with temporal shift out_t
        x_u = data[:, :duration, :, :].flatten(0, 1).unsqueeze(-1)  # (B, H, W, 1)
        y_u = data[:, out_t : out_t + duration, :, :].flatten(0, 1).unsqueeze(-1)  # (B, H, W, 1)

        b = int(x_u.shape[0])

        # append normalized grid in [0,1]
        grid_x = torch.linspace(0.0, 1.0, h, dtype=torch.float32).view(1, h, 1, 1).repeat(b, 1, w, 1)
        grid_y = torch.linspace(0.0, 1.0, w, dtype=torch.float32).view(1, 1, w, 1).repeat(b, h, 1, 1)

        x = torch.cat([grid_x, grid_y, x_u], dim=-1)  # (B, H, W, 3)
        y = y_u  # (B, H, W, 1)

        return x, y, x_normalizer, y_normalizer
