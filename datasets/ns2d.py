# datasets/ns_2d.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import scipy.io as sio
from h5py import File, Dataset

from .base import BaseDataset
from utils import UnitGaussianNormalizer, GaussianNormalizer


class NS2DDataset(BaseDataset):
    """
    2D Navier–Stokes dataset.

    Raw file is expected to contain key 'u':
      - For .mat: raw_data['u'] of shape (N, H, W, T)
      - For .h5:  raw_data['u'] of shape (H, W, T, N) and then transposed.
    """

    def __init__(self, data_args, **kwargs: Any) -> None:
        # Extra NS-specific options
        self.sample_factor: int = data_args.get("sample_factor", 2)
        self.normalize: bool = data_args.get("normalize", True)
        self.normalizer_type: str = data_args.get("normalizer_type", "PGN")

        super().__init__(data_args, **kwargs)

    # ------------ override hooks from BaseDataset ------------
    def load_raw_data(self, **kwargs: Any) -> torch.Tensor:
        """
        Load raw Navier–Stokes data as a torch.Tensor of shape (N, H, W, T).
        """
        data_path = self.data_path
        try:
            raw_data = sio.loadmat(data_path)
            data = torch.tensor(raw_data["u"], dtype=torch.float32)
        except Exception:
            with File(data_path, "r") as raw_data:
                u_node = raw_data["u"]
                if not isinstance(u_node, Dataset):
                    raise TypeError("Expected dataset 'u' in HDF5 file.")
                u_array = np.asarray(u_node[()])
            # HDF5 layout: (H, W, T, N) -> (N, H, W, T)
            data = torch.tensor(
                np.transpose(u_array, (3, 1, 2, 0)),
                dtype=torch.float32,
            )
        return data

    def process_split(
        self,
        data_split: torch.Tensor,
        mode: str,
        x_normalizer: Optional[torch.nn.Module] = None,
        y_normalizer: Optional[torch.nn.Module] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """
        Pre-process a split block:

        Args:
            data_split: torch.Tensor of shape (N, H, W, T)
            mode: one of 'train', 'valid', 'test'
            x_normalizer: optional normalizer for inputs (used in valid/test modes) 
            y_normalizer: optional normalizer for outputs (used in valid/test modes)
        Returns:
            x: input tensor of shape (B-1, H, W, C_in)
            y: output tensor of shape (B-1, H, W, C_out)
            x_normalizer: normalizer used for inputs
            y_normalizer: normalizer used for outputs
        """
        # raw: (N, H, W, T) -> (N, T, H, W) -> (B, H, W, C=1)
        in_t = self.data_args.get("in_t", 5)
        out_t = self.data_args.get("out_t", 1)
        duration = self.data_args.get("duration", 10)
        data = data_split[..., in_t: in_t + duration + out_t]
        data = data.permute(0, 3, 1, 2) # (N, T, H, W)
        
        N, T, H, W = data.shape

        if self.normalize:
            data = data.reshape(N * T, -1, 1)
            if mode == "train":
                if self.normalizer_type == "PGN":
                    x_normalizer = UnitGaussianNormalizer(data)
                    y_normalizer = x_normalizer
                else:
                    x_normalizer = GaussianNormalizer(data)
                    y_normalizer = x_normalizer
            else:
                if x_normalizer is None or y_normalizer is None:
                    raise RuntimeError(
                        "Normalizer is None in non-train mode with normalize=True"
                    )
            data = x_normalizer.encode(data).reshape(N, T, H, W)
        
        x = data[:, :duration, :, :].flatten(0, 1).unsqueeze(-1)  # (B, H, W, C_in)
        y = data[:, out_t: duration + out_t, :, :].flatten(0, 1).unsqueeze(-1)  # (B, H, W, C_out)
        
        B = x.shape[0]
        
        grid_x = torch.linspace(0, 1, H)
        grid_x = grid_x.reshape(1, H, 1, 1).repeat(B, 1, W, 1)
        grid_y = torch.linspace(0, 1, W)
        grid_y = grid_y.reshape(1, 1, W, 1).repeat(B, H, 1, 1)
        
        x = torch.cat([grid_x, grid_y, x], dim=-1) 

        return x, y, x_normalizer, y_normalizer
