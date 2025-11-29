# models/m2no/m2no_2d.py
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from .grid_operator import LPFOperator2d


class GridBlock2d(nn.Module):
    """
    Simple iterative grid smoother for a linear operator A(u) = f.

    Each block stores a list of 3x3 convolution smoothers S[i] and performs:
        u_{k+1} = u_k + S[i](f - A(u_k))
    starting from either:
        - u = S[0](f) if u is None, or
        - the provided initial guess u.

    Args:
        in_channels:  number of input channels
        out_channels: number of output channels
        num_iter:     number of smoothing iterations
        bias:         bias for Conv2d
        padding_mode: padding mode for Conv2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_iter: int,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iter = num_iter

        self.smoothers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    padding_mode=padding_mode,
                )
                for _ in range(num_iter)
            ]
        )

    def forward(
        self,
        A: nn.Module,
        f: Tensor,
        u: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            A: linear operator, typically a Conv2d (applied as A(u))
            f: right-hand side, shape (B, C, H, W)
            u: optional initial guess, same shape as f

        Returns:
            u: updated solution
            r: final residual r = f - A(u)
        """
        for i in range(self.num_iter):
            if u is None:
                # first sweep: initialize from f only
                u = self.smoothers[i](f)
            else:
                # subsequent sweeps: classic Richardson / Jacobi style step
                u = u + self.smoothers[i](f - A(u))

        if u is None:
            u = torch.zeros_like(f)

        r = f - A(u)
        return u, r


class MultiGrid2d(nn.Module):
    """
    Simple 2D multigrid V-cycle built on top of GridBlock2d and an LPF operator.

    - pre_S:  pre-smoothing at the finest scale
    - post_S: post-smoothing at the finest scale
    - grid_list[i]: smoother at level i (coarser grids)

    The LPF operator `op` provides:
        - restrict(x):   downsample to coarser grid
        - prolongate(x): upsample to finer grid
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_levels: List[int],
        op: LPFOperator2d,
        bias: bool = True,
        padding_mode: str = "zeros",
        resolutions: List[int] | Tuple[int, int] = (64, 64),
        norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_levels = grid_levels
        self.num_level = len(grid_levels)
        self.norm = norm

        # coarse linear operator A(u)
        self.A = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            padding_mode=padding_mode,
        )

        # pre-/post-smoothers at finest grid
        self.pre_S = GridBlock2d(in_channels, out_channels, num_iter=1, bias=bias, padding_mode=padding_mode)
        self.post_S = GridBlock2d(in_channels, out_channels, num_iter=1, bias=bias, padding_mode=padding_mode)

        # level-dependent smoothers on coarser grids
        self.grid_list = nn.ModuleList(
            [
                GridBlock2d(in_channels, out_channels, num_iter=grid_levels[i], bias=bias, padding_mode=padding_mode)
                for i in range(self.num_level)
            ]
        )

        # optional LayerNorm at each grid level
        self.resolutions = list(resolutions)
        if self.norm:
            self.norm_list = nn.ModuleList(
                [
                    nn.LayerNorm(
                        [out_channels, self.resolutions[0] // (2**i), self.resolutions[1] // (2**i)]
                    )
                    for i in range(self.num_level)
                ]
            )

        self.op = op

    def forward(self, f: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            f: RHS / input on finest grid, shape (B, C, H, W)

        Returns:
            u_n: final solution at finest grid
            r_n: final residual at finest grid
        """
        B, C, H_in, W_in = f.shape
        target_H, target_W = self.resolutions

        u_list: list[Optional[Tensor]] = [None] * (self.num_level + 1)
        r_list: list[Optional[Tensor]] = [None] * (self.num_level + 1)

        # pre-smoothing at finest grid
        u_n, r_n = self.pre_S(self.A, f)

        # bring (u, r) to the "reference" multigrid base resolution
        while u_n.shape[-2] > target_H:
            r_n = self.op.restrict(r_n)
            u_n = self.op.restrict(u_n)

        while u_n.shape[-2] < target_H:
            r_n = self.op.prolongate(r_n)
            u_n = self.op.prolongate(u_n)

        r_list[0] = r_n
        u_list[0] = u_n

        # go down to coarser levels
        for i in range(self.num_level):
            u_current = u_list[i]
            if u_current is None:
                raise ValueError(f"Missing state tensor at multigrid level {i}")
            u_coarse = self.op.restrict(u_current)
            u_coarse, r_coarse = self.grid_list[i](self.A, u_coarse)
            r_list[i + 1] = r_coarse
            u_list[i + 1] = u_coarse

        # go back up, adding corrections
        for i in range(self.num_level, 0, -1):
            u_coarse = u_list[i]
            if u_coarse is None:
                raise ValueError(f"Missing state tensor at multigrid level {i}")
            up = self.op.prolongate(u_coarse)
            prev_level = u_list[i - 1]
            if prev_level is None:
                raise ValueError(f"Missing state tensor at multigrid level {i - 1}")
            if self.norm:
                u_list[i - 1] = self.norm_list[i - 1](prev_level + up)
            else:
                u_list[i - 1] = prev_level + up

        u_n = u_list[0]
        if u_n is None:
            raise ValueError("Missing state tensor at multigrid level 0")

        # match the original input resolution
        while u_n.shape[-2] > H_in:
            u_n = self.op.restrict(u_n)

        while u_n.shape[-2] < H_in:
            u_n = self.op.prolongate(u_n)

        # post-smoothing at finest grid
        u_n, r_n = self.post_S(self.A, f, u_n)

        return u_n, r_n


class M2NO2d(nn.Module):
    """
    Multiwavelet-based Multigrid Neural Operator (M2NO) in 2D.

    Expected input / output:
        x: (B, H, W, C_in)
        y: (B, H, W, C_out)

    The pipeline is:
        x            -> Linear (C_in -> hidden) -> (B, hidden, H, W)
        multigrid V-cycles in feature space     -> (B, hidden, H, W)
        permute back -> MLP (hidden -> hidden)  -> Linear (hidden -> C_out)
    """

    def __init__(self, model_params: dict, **kwargs) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Read hyperparameters from model_params
        # ------------------------------------------------------------------
        self.in_channels: int = model_params.get("in_channels", 1)
        self.out_channels: int = model_params.get("out_channels", 1)
        k: int = model_params.get("k", 2)  # polynomial order per dimension
        c: int = model_params.get("c", 4)  # number of channels per polynomial
        self.num_layers: int = model_params.get("num_layers", 2)
        grid_levels: list[int] = model_params.get("grid_levels", [1, 1])
        base: str = model_params.get("base", "legendre")
        bias: bool = model_params.get("bias", True)
        padding_mode: str = model_params.get("padding_mode", "zeros")
        norm: bool = model_params.get("norm", False)
        resolutions = model_params.get("resolutions", [64, 64])
        activation_name: str = model_params.get("activation", "gelu")

        hidden_channels: int = c * (k**2)

        # ------------------------------------------------------------------
        # Linear projection in/out of hidden feature space
        # ------------------------------------------------------------------
        self.input_proj = nn.Linear(self.in_channels, hidden_channels)
        self.hidden_proj = nn.Linear(hidden_channels, hidden_channels)
        self.output_proj = nn.Linear(hidden_channels, self.out_channels)

        # ------------------------------------------------------------------
        # Low-pass / wavelet operator providing restrict / prolongate
        # ------------------------------------------------------------------
        self.wavelet_operator = LPFOperator2d(
            k=k,
            c=c,
            base=base,
        )

        # ------------------------------------------------------------------
        # Multigrid blocks stacked in depth
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                MultiGrid2d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    grid_levels=grid_levels,
                    op=self.wavelet_operator,
                    bias=bias,
                    padding_mode=padding_mode,
                    resolutions=resolutions,
                    norm=norm,
                )
                for _ in range(self.num_layers)
            ]
        )

        # ------------------------------------------------------------------
        # Nonlinearity
        # ------------------------------------------------------------------
        if activation_name == "gelu":
            self.activation = nn.GELU()
        elif activation_name == "relu":
            self.activation = nn.ReLU()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input tensor, shape (B, H, W, C_in)

        Returns:
            y: output tensor, shape (B, H, W, C_out)
        """
        assert x.dim() == 4, f"Expected x of shape (B, H, W, C_in), got {x.shape}"

        B, H, W, C_in = x.shape
        if C_in != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: got {C_in}, expected {self.in_channels}"
            )

        # (B, H, W, C_in) -> (B, H, W, hidden)
        u = self.input_proj(x)

        # (B, H, W, hidden) -> (B, hidden, H, W)
        u = u.permute(0, 3, 1, 2).contiguous()

        # Multigrid V-cycles
        for i, block in enumerate(self.blocks):
            u, _ = block(u)
            if i < self.num_layers - 1:
                u = self.activation(u)

        # (B, hidden, H, W) -> (B, H, W, hidden)
        u = u.permute(0, 2, 3, 1).contiguous()

        # pointwise MLP in hidden space
        u = self.activation(self.hidden_proj(u))
        y = self.output_proj(u)

        return y
