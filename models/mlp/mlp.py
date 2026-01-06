# models/mlp.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """
    Point-wise MLP for PDE fields on arbitrary geometries.

    Expected input:
        x:      (B, N, C_in)        field features (for NS2D: C_in = 1)
        coords: optional geometric coordinates:
                - (N, C_coord)
                - (1, N, C_coord)
                - (B, N, C_coord)

    The model first concatenates features and coords (if enabled), then applies
    a shared MLP to each point independently:

        [field, coords] -> hidden_dims -> out_channels

    All geometry handling is external: dataset provides `geom` and `coords`,
    trainer/forecaster passes them into `forward`.
    """

    def __init__(self, model_params: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__()

        # Core dims
        self.in_channels: int = int(model_params.get("in_channels", 1))
        self.out_channels: int = int(model_params.get("out_channels", 1))

        # Geometry / coordinates
        self.use_coords: bool = bool(model_params.get("use_coords", True))
        self.coord_dim: int = int(model_params.get("coord_dim", 0))  # e.g. 2 for 2D grid, 3 for 3D, etc.

        # MLP structure
        hidden_dims: List[int] = list(model_params.get("hidden_dims", [128, 128, 128]))
        activation_name: str = str(model_params.get("activation", "gelu"))
        self.dropout_p: float = float(model_params.get("dropout", 0.0))
        self.use_residual: bool = bool(model_params.get("use_residual", True))

        act = _get_activation(activation_name)

        in_dim = self.in_channels + (self.coord_dim if self.use_coords else 0)

        dims: List[int] = [in_dim] + hidden_dims + [self.out_channels]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))

        # Last layer without activation
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _broadcast_coords(
        coords: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Normalize coords shape to (B, N, C_coord).

        Supported input shapes:
            (N, C_coord)
            (1, N, C_coord)
            (B, N, C_coord)

        Returns:
            coords_b: (B, N, C_coord)
        """
        if coords.ndim == 2:
            # (N, C_coord) -> (1, N, C_coord) -> (B, N, C_coord)
            coords = coords.unsqueeze(0)
        if coords.ndim != 3:
            raise ValueError(f"coords must be 2D or 3D, got shape {tuple(coords.shape)}")

        if coords.shape[0] == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1)
        elif coords.shape[0] != batch_size:
            raise ValueError(
                f"Incompatible coords batch size: coords.shape[0]={coords.shape[0]}, "
                f"expected batch_size={batch_size}."
            )
        return coords

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, N, C_in)
            coords: optional, (N, C_coord) or (B, N, C_coord)
            geom:   optional geometry dict (not used directly by this MLP,
                    but kept for a unified model interface).

        Returns:
            y: (B, N, out_channels)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x of shape (B, N, C), got {tuple(x.shape)}")

        b, n, c_in = x.shape
        if c_in != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: got {c_in}, expected {self.in_channels}"
            )

        features = x

        if self.use_coords:
            if coords is None:
                raise ValueError("use_coords=True but coords is None.")
            coords_b = self._broadcast_coords(coords, batch_size=b)  # (B, N, C_coord)
            if coords_b.shape[-1] != self.coord_dim:
                raise ValueError(
                    f"coord_dim mismatch: got {coords_b.shape[-1]}, expected {self.coord_dim}"
                )
            features = torch.cat([features, coords_b], dim=-1)  # (B, N, C_in + C_coord)

        b, n, c = features.shape
        features_flat = features.view(b * n, c)  # (B*N, C_total)

        out_flat = self.mlp(features_flat)       # (B*N, out_channels)
        out = out_flat.view(b, n, self.out_channels)

        # Optional residual connection on the field channel, if shapes match
        if self.use_residual and self.out_channels == self.in_channels:
            out = out + x

        return out
