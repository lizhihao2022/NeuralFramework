# models/fno/fno.py
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_spatial
from .spectral_conv import SpectralConv1d, SpectralConv2d, SpectralConv3d


class FNO1d(nn.Module):
    """
    Fourier Neural Operator for 1D fields.

    Interface:
        input:  x in shape (B, L, C_in)
        output: y in shape (B, L, C_out)

    Typical model_params:
        {
            "in_channels": 3,
            "out_channels": 1,
            "modes": 16,
            "width": 64,
            "n_layers": 4,
            "padding": 0,       # optional spatial padding on the right
            "use_grid": True,   # whether to append 1D coordinate
            "fc_dim": 128,      # hidden dim of final pointwise MLP (optional)
        }
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Core config
        self.in_channels: int = model_params.get("in_channels", 3)
        self.out_channels: int = model_params.get("out_channels", 1)
        self.modes: int = model_params.get("modes", 16)
        self.width: int = model_params.get("width", 64)
        self.n_layers: int = model_params.get("n_layers", 4)
        self.padding: int = model_params.get("padding", 0)
        self.use_grid: bool = model_params.get("use_grid", True)

        # Final "fc" hidden dim (per-point MLP)
        self.fc_dim: int = model_params.get("fc_dim", 2 * self.width)

        # If use_grid=True, append 1 coordinate channel (x ∈ [0, 1])
        in_proj_in_channels = self.in_channels + (1 if self.use_grid else 0)

        # Lift to width channels via 1×1 conv
        self.input_proj = nn.Conv1d(
            in_channels=in_proj_in_channels,
            out_channels=self.width,
            kernel_size=1,
        )

        # Spectral + local (1×1) paths for each layer
        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv1d(self.width, self.width, self.modes)
                for _ in range(self.n_layers)
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size=1)
                for _ in range(self.n_layers)
            ]
        )

        self.activation = nn.GELU()

        # Final pointwise MLP (per grid point), implemented as 1×1 convs
        # fc1: width -> fc_dim
        # fc2: fc_dim -> out_channels
        self.output_layers = nn.Sequential(
            nn.Conv1d(self.width, self.fc_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.fc_dim, self.out_channels, kernel_size=1),
        )

    @staticmethod
    def _make_grid(size: int, device: torch.device) -> torch.Tensor:
        """
        Build a 1D coordinate grid in [0, 1] with shape (1, 1, L).
        """
        grid = torch.linspace(0.0, 1.0, size, device=device)
        grid = grid.view(1, 1, size)
        return grid

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, C_in), channels-last.

        Returns:
            Tensor of shape (B, L, C_out), channels-last.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, L, C), got {x.shape}")

        B, L, C_in = x.shape
        if C_in != self.in_channels:
            raise ValueError(
                f"Input channels {C_in} != configured in_channels {self.in_channels}"
            )

        # (B, L, C_in) -> (B, C_in, L)
        x = x.permute(0, 2, 1).contiguous()

        # Optional padding on the spatial dimension (right side)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))
            L_p = L + self.padding
        else:
            L_p = L

        # Append coordinate grid if requested
        if self.use_grid:
            grid = self._make_grid(L_p, x.device)   # (1, 1, L_p)
            grid = grid.expand(B, -1, -1)           # (B, 1, L_p)
            x = torch.cat([x, grid], dim=1)         # (B, C_in+1, L_p)

        # Lift to width channels
        x = self.input_proj(x)  # (B, width, L_p)

        # Spectral blocks
        for spec_conv, w in zip(self.spectral_convs, self.ws):
            x1 = spec_conv(x)   # spectral branch
            x2 = w(x)           # local branch (1x1 conv)
            x = self.activation(x1 + x2)

        # Final per-point MLP
        x = self.output_layers(x)  # (B, C_out, L_p)

        # Remove padding
        if self.padding > 0:
            x = x[..., :L]  # (B, C_out, L)

        # Back to channels-last
        x = x.permute(0, 2, 1).contiguous()  # (B, L, C_out)
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator (FNO) with channels-last PDE interface.

    Design:
      - Optionally augment input with normalized spatial coordinates (y, x) ∈ [0,1]^2.
      - Lift to a higher-dimensional channel space (width).
      - Apply several spectral convolution blocks:
            x_{k+1} = GELU( SpectralConv2d(x_k) + 1x1Conv(x_k) )
      - Apply final per-point MLP to map width → fc_dim → out_channels.

    Interface:
      forward(x): x ∈ R^{B × H × W × C_in} → y ∈ R^{B × H × W × C_out}

    Expected model_params:
      - in_channels (int, required)
      - out_channels (int, required)
      - modes_x (int, default=12)
      - modes_y (int, default=12)
      - width (int, default=64)
      - n_layers (int, default=4)
      - padding (int, default=0)      # pad on right / bottom
      - use_grid (bool, default=True) # append (y, x)
      - fc_dim (int, optional)        # hidden dim of final MLP, default 2*width
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # ---- Core configuration ----
        in_channels: int = model_params.get("in_channels", 1)
        out_channels: int = model_params.get("out_channels", 1)

        modes_x: int = model_params.get("modes_x", 12)
        modes_y: int = model_params.get("modes_y", 12)
        width: int = model_params.get("width", 64)
        n_layers: int = model_params.get("n_layers", 4)

        padding: int = model_params.get("padding", 0)
        use_grid: bool = model_params.get("use_grid", True)
        fc_dim: int = model_params.get("fc_dim", 2 * width)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        self.padding = padding
        self.use_grid = use_grid
        self.fc_dim = fc_dim

        # First projection: from (C_in (+ 2 coords)) → width
        in_proj_in_channels = in_channels + (2 if use_grid else 0)
        self.input_proj = nn.Conv2d(in_proj_in_channels, width, kernel_size=1)

        # Spectral blocks
        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x, modes_y) for _ in range(n_layers)]
        )
        # Local 1x1 conv in physical space for each block
        self.ws = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        self.activation = nn.GELU()

        # Final projection: per-point MLP width → fc_dim → out_channels
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, fc_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fc_dim, out_channels, kernel_size=1),
        )

    @staticmethod
    def _make_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Construct a normalized coordinate grid in [0, 1]^2.

        Returns:
            grid: Tensor of shape (1, 2, H, W)
                  grid[:, 0, :, :] = y / (H - 1)
                  grid[:, 1, :, :] = x / (W - 1)
        """
        y = torch.linspace(0.0, 1.0, H, device=device)
        x = torch.linspace(0.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((grid_y, grid_x), dim=0)  # (2, H, W)
        return grid.unsqueeze(0)  # (1, 2, H, W)

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, H, W, C_in), channels-last.
            coords: Optional tensor of coordinates.
            geom: Optional geometry dictionary.

        Returns:
            Tensor of shape (B, H, W, C_out), channels-last.
        """
        x, coords, ctx = to_spatial(x, coords=coords, geom=geom, require_shape=False)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, H, W, C), got {x.shape}")

        B, H, W, C = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {C}"
            )

        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Optional zero-padding on right / bottom
        if self.padding > 0:
            pad = self.padding
            # pad format: (pad_left, pad_right, pad_top, pad_bottom)
            x = F.pad(x, (0, pad, 0, pad))
            H_p, W_p = H + pad, W + pad
        else:
            H_p, W_p = H, W

        # Append coordinate grid channels if requested
        if self.use_grid:
            grid = self._make_grid(H_p, W_p, x.device)  # (1, 2, H_p, W_p)
            grid = grid.expand(B, -1, -1, -1)           # (B, 2, H_p, W_p)
            x = torch.cat([x, grid], dim=1)             # (B, C_in+2, H_p, W_p)

        # Lift to width channels
        x = self.input_proj(x)  # (B, width, H_p, W_p)

        # Spectral blocks
        for spec_conv, w in zip(self.spectral_convs, self.ws):
            x1 = spec_conv(x)   # spectral conv
            x2 = w(x)           # local 1x1 conv
            x = self.activation(x1 + x2)

        # Final pointwise MLP
        x = self.output_proj(x)  # (B, C_out, H_p, W_p)

        # Remove padding if applied
        if self.padding > 0:
            x = x[:, :, :H, :W]

        # Back to channels-last
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C_out)
        
        x = ctx.restore_x(x)
        
        return x


class FNO3d(nn.Module):
    """
    Fourier Neural Operator for 3D fields.

    Interface:
        input:  x in shape (B, D, H, W, C_in)
        output: y in shape (B, D, H, W, C_out)

    D, H, W can be (z, y, x), or (time, y, x), etc., depending on the PDE.

    Typical model_params:
        {
            "in_channels": 3,
            "out_channels": 1,
            "modes1": 12,
            "modes2": 12,
            "modes3": 12,
            "width": 32,
            "n_layers": 4,
            "padding": (0, 0, 0),   # or int
            "use_grid": True,
            "fc_dim": 128,          # hidden dim of final MLP (optional)
        }
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.in_channels: int = model_params.get("in_channels", 3)
        self.out_channels: int = model_params.get("out_channels", 1)

        self.modes1: int = model_params.get("modes1", 12)
        self.modes2: int = model_params.get("modes2", 12)
        self.modes3: int = model_params.get("modes3", 12)

        self.width: int = model_params.get("width", 32)
        self.n_layers: int = model_params.get("n_layers", 4)

        padding = model_params.get("padding", (0, 0, 0))
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding: Tuple[int, int, int] = padding

        self.use_grid: bool = model_params.get("use_grid", True)
        self.fc_dim: int = model_params.get("fc_dim", 2 * self.width)

        # Append 3 coordinate channels (d, h, w) if use_grid=True
        in_proj_in_channels = self.in_channels + (3 if self.use_grid else 0)

        # Lift to `width` channels via 1×1 conv
        self.input_proj = nn.Conv3d(
            in_channels=in_proj_in_channels,
            out_channels=self.width,
            kernel_size=1,
        )

        # Spectral + local paths for each layer
        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv3d(
                    self.width,
                    self.width,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv3d(self.width, self.width, kernel_size=1)
                for _ in range(self.n_layers)
            ]
        )

        self.activation = nn.GELU()

        # Final per-point MLP: width -> fc_dim -> out_channels
        # 对应你给的参考代码里的 fc1 / fc2，只是用 Conv3d(kernel_size=1) 实现
        self.output_layers = nn.Sequential(
            nn.Conv3d(self.width, self.fc_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.fc_dim, self.out_channels, kernel_size=1),
        )

    @staticmethod
    def _make_grid(D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Build a 3D coordinate grid in [0, 1]^3 with shape (1, 3, D, H, W).
        """
        d = torch.linspace(0.0, 1.0, D, device=device)
        h = torch.linspace(0.0, 1.0, H, device=device)
        w = torch.linspace(0.0, 1.0, W, device=device)

        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")
        grid = torch.stack((grid_d, grid_h, grid_w), dim=0)  # (3, D, H, W)
        return grid.unsqueeze(0)  # (1, 3, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D, H, W, C_in), channels-last.

        Returns:
            y: Tensor of shape (B, D, H, W, C_out), channels-last.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, D, H, W, C), got {x.shape}")
        B, D, H, W, C_in = x.shape
        if C_in != self.in_channels:
            raise ValueError(
                f"Input channels {C_in} != configured in_channels {self.in_channels}"
            )

        # channels-last -> channels-first
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C_in, D, H, W)

        pad_d, pad_h, pad_w = self.padding
        if pad_d or pad_h or pad_w:
            # pad order for 5D (N, C, D, H, W):
            # (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D_p, H_p, W_p = D + pad_d, H + pad_h, W + pad_w
        else:
            D_p, H_p, W_p = D, H, W

        # Append coordinate grid if requested
        if self.use_grid:
            grid = self._make_grid(D_p, H_p, W_p, x.device)  # (1, 3, D_p, H_p, W_p)
            grid = grid.expand(B, -1, -1, -1, -1)
            x = torch.cat([x, grid], dim=1)  # (B, C_in+3, D_p, H_p, W_p)

        # Lift to width channels
        x = self.input_proj(x)  # (B, width, D_p, H_p, W_p)

        # Spectral blocks
        for spec_conv, w in zip(self.spectral_convs, self.ws):
            x1 = spec_conv(x)
            x2 = w(x)
            x = self.activation(x1 + x2)

        # Final per-point MLP
        x = self.output_layers(x)  # (B, C_out, D_p, H_p, W_p)

        # Remove padding if any
        if pad_d or pad_h or pad_w:
            x = x[:, :, :D, :H, :W]

        # channels-first -> channels-last
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C_out)
        return x
