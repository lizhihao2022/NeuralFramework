# models/m2no/grid_operator.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor

from .utils import get_filter


class LPFOperator1d(nn.Module):
    """
    1D low-pass multiwavelet operator providing restrict / prolongate
    along a single spatial dimension.

    Channel layout:
        - total channels: C = c * fine_dim
        - fine_dim = 2 * k  (number of fine-scale modal coefficients)
        - coarse_dim = k    (number of coarse-scale modal coefficients)

    The internal filter H has shape (coarse_dim, fine_dim).
    """

    def __init__(
        self,
        c: int = 4,
        k: int = 4,
        base: str = "legendre",
        **kwargs,
    ) -> None:
        super().__init__()
        self.c = c
        self.k = k
        self.fine_dim = 2 * k
        self.coarse_dim = k
        self.sub_dim = self.coarse_dim  # kept for compatibility

        H, G = self._build_filter(base, k)
        # register as buffers so they follow .to(device)
        self.register_buffer("H", H)
        self.register_buffer("G", G)  # not used yet, but kept for high-pass if needed

    @staticmethod
    def _build_filter(base: str, k: int) -> Tuple[Tensor, Tensor]:
        """
        Build 1D low-pass / high-pass filter matrices H, G from the
        polynomial filter generator.
        """
        H0, H1, G0, G1, _, _ = get_filter(base, k)

        H_np = np.concatenate((H0, H1), axis=1)  # (k, 2k)
        G_np = np.concatenate((G0, G1), axis=1)  # (k, 2k)

        H = torch.tensor(H_np, dtype=torch.float32)
        G = torch.tensor(G_np, dtype=torch.float32)
        return H, G

    def restrict(self, x: Tensor) -> Tensor:
        """
        Restrict (downsample) along the last spatial dimension by factor 2.

        Args:
            x: tensor of shape (B, C, N), where
               C = c * fine_dim (typically c * 2k),
               N is the fine spatial resolution.

        Returns:
            Tensor of shape (B, C_out, N_out),
            where C_out = c * coarse_dim, N_out = N // 2.
        """
        B, C, N = x.shape
        # (B, C, N) -> (B, N, C)
        x_ = x.permute(0, 2, 1).contiguous()
        # (B, N/2, c, fine_dim)
        x_ = x_.view(B, N // 2, self.c, -1)
        # (B, N/2, c, coarse_dim)
        x_ = torch.matmul(x_, self.H.t())
        # (B, N/2, c * coarse_dim)
        x_ = x_.view(B, N // 2, -1)
        # (B, C_out, N/2)
        x_ = x_.permute(0, 2, 1).contiguous()
        return x_

    def prolongate(self, x: Tensor) -> Tensor:
        """
        Prolongate (upsample) along the last spatial dimension by factor 2.

        Args:
            x: tensor of shape (B, C_in, N), where
               C_in = c * coarse_dim.

        Returns:
            Tensor of shape (B, C_out, 2N),
            where C_out = c * fine_dim.
        """
        B, C, N = x.shape
        # (B, C_in, N) -> (B, N, C_in)
        x_ = x.permute(0, 2, 1).contiguous()
        # (B, N, c, coarse_dim)
        x_ = x_.view(B, N, self.c, -1)
        # (B, N, c, fine_dim)
        x_ = torch.matmul(x_, self.H)
        # (B, 2N, c * fine_dim)
        x_ = x_.view(B, N * 2, -1)
        # (B, C_out, 2N)
        x_ = x_.permute(0, 2, 1).contiguous()
        return x_


class LPFOperator2d(nn.Module):
    """
    2D low-pass multiwavelet operator providing restrict / prolongate
    on a 2D grid (H, W) with channels encoded in modal space.

    For a given (k, c):
        - hidden_channel is typically c * (k^2) or c * (4 * k^2),
          depending on how you pack subbands.
        - internally, H acts on a "fine_dim" and maps it to a "coarse_dim".

    The current implementation keeps the original channel layout used in
    the M2NO code and only refactors it for style / type checking.
    """

    def __init__(
        self,
        k: int = 4,
        c: int = 4,
        base: str = "legendre",
        **kwargs,
    ) -> None:
        super().__init__()
        self.c = c
        self.k = k

        # These three attributes are kept for backward compatibility.
        # Note: depending on how you pack the modes, hidden_channel
        # may not be exactly "channels" in the input tensor.
        self.hidden_channel = c * (k**2)
        self.fine_dim = self.hidden_channel
        self.coarse_dim = self.hidden_channel // 4
        self.sub_dim = self.coarse_dim * 3

        H, G = self._build_filter(base, k)
        self.register_buffer("H", H)
        self.register_buffer("G", G)

    @staticmethod
    def _build_filter(base: str, k: int) -> Tuple[Tensor, Tensor]:
        """
        Build 2D low-pass / high-pass filter matrices H, G using Kronecker
        products of 1D filters.
        """
        H0, H1, G0, G1, _, _ = get_filter(base, k)

        H0 = np.asarray(H0)
        H1 = np.asarray(H1)
        G0 = np.asarray(G0)
        G1 = np.asarray(G1)

        # Low-pass (scaling) part: four combinations
        H_LL = np.kron(H0, H0)
        H_LH = np.kron(H0, H1)
        H_HL = np.kron(H1, H0)
        H_HH = np.kron(H1, H1)
        H_np = np.concatenate((H_LL, H_LH, H_HL, H_HH), axis=1)

        # High-pass combinations (G*H, H*G, G*G)
        GH_LL = np.kron(G0, H0)
        GH_LH = np.kron(G0, H1)
        GH_HL = np.kron(G1, H0)
        GH_HH = np.kron(G1, H1)
        GH = np.concatenate((GH_LL, GH_LH, GH_HL, GH_HH), axis=1)

        HG_LL = np.kron(H0, G0)
        HG_LH = np.kron(H0, G1)
        HG_HL = np.kron(H1, G0)
        HG_HH = np.kron(H1, G1)
        HG = np.concatenate((HG_LL, HG_LH, HG_HL, HG_HH), axis=1)

        GG_LL = np.kron(G0, G0)
        GG_LH = np.kron(G0, G1)
        GG_HL = np.kron(G1, G0)
        GG_HH = np.kron(G1, G1)
        GG = np.concatenate((GG_LL, GG_LH, GG_HL, GG_HH), axis=1)

        H = torch.tensor(H_np, dtype=torch.float32)
        G = torch.tensor(np.concatenate((GH, HG, GG), axis=0), dtype=torch.float32)
        return H, G

    def restrict(self, x: Tensor) -> Tensor:
        """
        Restrict (downsample) by factor 2 in both H and W.

        Args:
            x: tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, C_out, H/2, W/2).
        """
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, H, W, C)
        x_ = x.permute(0, 2, 3, 1).contiguous()
        # (B, H/2, W/2, c, fine_dim_per_group)
        x_ = x_.view(B, H // 2, W // 2, self.c, -1)
        # apply low-pass filter along the modal dimension
        # (B, H/2, W/2, c, coarse_dim_per_group)
        x_ = torch.matmul(x_, self.H.t())
        # (B, H/2, W/2, C_out)
        x_ = x_.view(B, H // 2, W // 2, -1)
        # (B, C_out, H/2, W/2)
        x_ = x_.permute(0, 3, 1, 2).contiguous()
        return x_

    def prolongate(self, x: Tensor) -> Tensor:
        """
        Prolongate (upsample) by factor 2 in both H and W.

        Args:
            x: tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, 2H, 2W).
        """
        B, C, H, W = x.shape

        # (B, C_in, H, W) -> (B, H, W, C_in)
        x_ = x.permute(0, 2, 3, 1).contiguous()
        # (B, H, W, c, coarse_dim_per_group)
        x_ = x_.view(B, H, W, self.c, -1)
        # (B, H, W, c, fine_dim_per_group)
        x_ = torch.matmul(x_, self.H)
        # (B, 2H, 2W, C_out)
        x_ = x_.view(B, H * 2, W * 2, -1)
        # (B, C_out, 2H, 2W)
        x_ = x_.permute(0, 3, 1, 2).contiguous()
        return x_
