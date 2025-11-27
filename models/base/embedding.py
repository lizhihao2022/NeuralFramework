# models/base/embedding.py
import math

from typing import Sequence, Optional, Tuple

import torch
from torch import nn, Tensor


def build_grid_coords(
    shape: Sequence[int],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    coord_min: float = 0.0,
    coord_max: float = 1.0,
) -> Tensor:
    """
    Build a regular grid of coordinates in [coord_min, coord_max] for each axis.

    Args:
        shape: Spatial shape for each dimension, e.g. (H, W) or (H, W, D).
        device: Target device for the coordinates.
        dtype: Data type for the coordinates.
        coord_min: Lower bound for each coordinate axis.
        coord_max: Upper bound for each coordinate axis.

    Returns:
        coords: Tensor of shape (N, D), where
                N = prod(shape), D = len(shape).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dims = len(shape)
    # 1D list of coordinates for each axis
    axes = [
        torch.linspace(coord_min, coord_max, steps=s, device=device, dtype=dtype)
        for s in shape
    ]
    # Create meshgrid: each element has shape (s1, s2, ..., sD)
    mesh = torch.meshgrid(*axes, indexing="ij")  # list of length D
    # Stack last dimension -> (s1, s2, ..., sD, D)
    grid = torch.stack(mesh, dim=-1)
    # Flatten spatial dims -> (N, D)
    coords = grid.reshape(-1, dims)
    return coords


def unified_pos_embedding(
    shape_list: Sequence[int],
    ref: int,
    *,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    coord_min: float = 0.0,
    coord_max: float = 1.0,
) -> Tensor:
    """
    Build a unified positional distance embedding between an input grid
    and a reference grid, for arbitrary spatial dimensions.

    This generalizes your original implementation for 1D/2D/3D to N-D.

    Args:
        shape_list: Spatial shape of the input grid, e.g.
                    [H] for 1D, [H, W] for 2D, [H, W, D] for 3D, etc.
        ref: Number of reference points per dimension
             (reference grid will have shape [ref, ref, ..., ref]).
        batch_size: Batch size (the distance matrix is simply broadcasted
                    along batch dimension).
        device: Target device for computations.
        dtype: Data type for coordinates and distances.
        coord_min: Lower bound for coordinate values (default 0.0).
        coord_max: Upper bound for coordinate values (default 1.0).

    Returns:
        pos: Positional distance tensor of shape (B, N, M), where
             B = batch_size,
             N = prod(shape_list)      (number of input grid points),
             M = ref ** len(shape_list) (number of reference grid points).
             Each entry pos[b, i, j] is the Euclidean distance between
             input point i and reference point j in coordinate space.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input grid coordinates: (N, D)
    coords_in = build_grid_coords(
        shape_list,
        device=device,
        dtype=dtype,
        coord_min=coord_min,
        coord_max=coord_max,
    )  # (N, D)

    # Reference grid coordinates: (M, D)
    dims = len(shape_list)
    ref_shape = [ref] * dims
    coords_ref = build_grid_coords(
        ref_shape,
        device=device,
        dtype=dtype,
        coord_min=coord_min,
        coord_max=coord_max,
    )  # (M, D)

    # Compute pairwise Euclidean distances
    # coords_in:  (N, D)
    # coords_ref: (M, D)
    # diff: (N, M, D)
    diff = coords_in.unsqueeze(1) - coords_ref.unsqueeze(0)
    # dist: (N, M)
    dist = torch.linalg.norm(diff, dim=-1)

    # Broadcast across batch: (B, N, M)
    pos = dist.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return pos


class RotaryEmbedding1D(nn.Module):
    """
    1D Rotary positional embedding (RoPE) for a single coordinate axis.

    Typical use:
        rope = RotaryEmbedding1D(dim=32)  # dim per axis (must be even)
        cos_x, sin_x = rope(x_coords)     # x_coords: (B, N)

    In attention:
        # q, k: (B, H, N, D), with D = D_axis * num_axes (e.g., num_axes=2 for 2D)
        cos_x = cos_x.unsqueeze(1)  # -> (B, 1, N, D_axis) to broadcast over heads
        sin_x = sin_x.unsqueeze(1)
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension per axis must be even."

        self.dim = dim
        self.base = base
        self.scale = scale

        # frequencies for half of the dimensions: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inv_freq = inv_freq * scale
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            coords: tensor of shape (..., N)
                    e.g. (B, N) physical coordinates along one axis (x / y / z / t).

        Returns:
            cos, sin: both of shape (..., N, dim)
                      ready to be broadcast to q/k whose last dim is `dim` (per axis).
        """
        # coords: (..., N)
        coords = coords.to(self.inv_freq.device).to(self.inv_freq.dtype)

        # angles: (..., N, dim/2)
        angles = torch.einsum("... n, d -> ... n d", coords, self.inv_freq)

        cos = angles.cos()
        sin = angles.sin()

        # duplicate each frequency so last dim = dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # (..., N, dim)
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # (..., N, dim)
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate last dimension by 90 degrees in each pair:
        (x1, x2) -> (-x2, x1)

    Args:
        x: (..., D), where D is even.

    Returns:
        rotated x: (..., D)
    """
    D = x.shape[-1]
    assert D % 2 == 0, "Last dimension must be even for rotate_half."

    # view as (..., D/2, 2)
    x = x.view(*x.shape[:-1], D // 2, 2)
    x1, x2 = x.unbind(dim=-1)  # (..., D/2)
    return torch.cat([-x2, x1], dim=-1)  # (..., D)


def rotary_pos_embedding(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply RoPE on the last dimension of x.

    Args:
        x:   (..., N, D)
        cos: (..., N, D) - from RotaryEmbedding1D
        sin: (..., N, D)

    Returns:
        x_out: (..., N, D)
    """
    return x * cos + rotate_half(x) * sin


def rotary_2d_pos_embedding(
    x: Tensor,
    cos_x: Tensor,
    sin_x: Tensor,
    cos_y: Tensor,
    sin_y: Tensor,
) -> Tensor:
    """
    2D RoPE: channels are split into x-axis part and y-axis part.

    Args:
        x:      (..., N, D)  e.g. (B, H, N, D) or (B, N, D)
        cos_x:  (..., N, D_axis)
        sin_x:  (..., N, D_axis)
        cos_y:  (..., N, D_axis)
        sin_y:  (..., N, D_axis)

    Assumes:
        D = 2 * D_axis, and D_axis is even.

    Returns:
        x_out: same shape as x
    """
    D = x.shape[-1]
    assert D % 2 == 0, "Embedding dim must be divisible by 2 for 2D RoPE."
    D_axis = D // 2
    assert D_axis % 2 == 0, "Each axis dim must be even for RoPE."

    x_x, x_y = x[..., :D_axis], x[..., D_axis:]

    x_x_out = rotary_pos_embedding(x_x, cos_x, sin_x)
    x_y_out = rotary_pos_embedding(x_y, cos_y, sin_y)

    return torch.cat([x_x_out, x_y_out], dim=-1)


def rotary_3d_pos_embedding(
    x: Tensor,
    cos_x: Tensor,
    sin_x: Tensor,
    cos_y: Tensor,
    sin_y: Tensor,
    cos_z: Tensor,
    sin_z: Tensor,
) -> Tensor:
    """
    3D RoPE: channels are split into x / y / z parts.

    Args:
        x:      (..., N, D)
        cos_x:  (..., N, D_axis)
        sin_x:  (..., N, D_axis)
        cos_y:  (..., N, D_axis)
        sin_y:  (..., N, D_axis)
        cos_z:  (..., N, D_axis)
        sin_z:  (..., N, D_axis)

    Assumes:
        D = 3 * D_axis, and D_axis is even.

    Returns:
        x_out: same shape as x
    """
    D = x.shape[-1]
    assert D % 3 == 0, "Embedding dim must be divisible by 3 for 3D RoPE."
    D_axis = D // 3
    assert D_axis % 2 == 0, "Each axis dim must be even for RoPE."

    x_x, x_y, x_z = x.split(D_axis, dim=-1)

    x_x_out = rotary_pos_embedding(x_x, cos_x, sin_x)
    x_y_out = rotary_pos_embedding(x_y, cos_y, sin_y)
    x_z_out = rotary_pos_embedding(x_z, cos_z, sin_z)

    return torch.cat([x_x_out, x_y_out, x_z_out], dim=-1)


def timestep_embedding(
    timesteps: Tensor,
    dim: int,
    max_period: float = 10000.0,
    repeat_only: bool = False,
) -> Tensor:
    """
    Create sinusoidal timestep embeddings (DDPM-style).

    Args:
        timesteps: 1D or ND tensor of timesteps, e.g. shape (N,) or (B,)
                   values can be integer steps or continuous time.
        dim:       output embedding dimension.
        max_period: controls the minimum frequency of the embeddings.
        repeat_only: if True, just repeat the raw timesteps (no sin/cos),
                     useful for ablation or very low-dimensional cases.

    Returns:
        emb: tensor of shape (..., dim), where ... is timesteps.shape
    """
    # ensure float
    timesteps = timesteps.float()

    if repeat_only:
        # simply repeat scalar timesteps to desired dimension
        while timesteps.ndim < 2:
            timesteps = timesteps.unsqueeze(-1)  # (..., 1)
        return timesteps.repeat_interleave(dim, dim=-1)

    half = dim // 2
    device = timesteps.device

    # frequencies: (half,)
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
        / half
    )

    # timesteps: (...,) -> (..., 1)
    args = timesteps.unsqueeze(-1) * freqs  # (..., half)

    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (..., 2*half)

    # if dim is odd, pad one zero channel
    if dim % 2 == 1:
        pad_shape = list(emb.shape[:-1]) + [1]
        emb = torch.cat([emb, emb.new_zeros(pad_shape)], dim=-1)

    return emb  # shape (..., dim)
