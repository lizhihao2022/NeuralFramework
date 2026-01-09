# utils/spatial.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def prod(shape: Sequence[int]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return int(p)


def resolve_shape(
    shape: Optional[Sequence[int]] = None,
    geom: Optional[Dict[str, Any]] = None,
    *,
    keys: Sequence[str] = ("shape", "spatial_shape", "grid_shape"),
) -> Optional[List[int]]:
    """
    Resolve spatial shape from (explicit shape) or geom dict.

    Priority:
      1) explicit `shape`
      2) geom[key] for key in keys
    """
    if shape is not None:
        shp = list(shape)
        return shp if len(shp) > 0 else None

    if geom is None:
        return None

    for k in keys:
        v = geom.get(k, None)
        if v is None:
            continue
        shp = list(v)
        return shp if len(shp) > 0 else None

    return None


# -------------------------
# Tensor layout predicates
# -------------------------
def is_bnc(x: torch.Tensor) -> bool:
    return x.ndim == 3


def is_bsnc(x: torch.Tensor) -> bool:
    return x.ndim == 4  # (B,S,N,C)


def is_bspatialc(x: torch.Tensor, shape: Sequence[int]) -> bool:
    # (B,*shape,C)
    if x.ndim != len(shape) + 2:
        return False
    return list(x.shape[1 : 1 + len(shape)]) == list(shape)


def is_bsspatialc(x: torch.Tensor, shape: Sequence[int]) -> bool:
    # (B,S,*shape,C)
    if x.ndim != len(shape) + 3:
        return False
    return list(x.shape[2 : 2 + len(shape)]) == list(shape)


# -------------------------
# Unflatten / flatten
# -------------------------
def unflatten_bnc(
    x: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    (B,N,C) -> (B,*shape,C)
    """
    if x.ndim != 3:
        raise ValueError(f"unflatten_bnc expects (B,N,C), got {tuple(x.shape)}")

    B, N, C = x.shape
    P = prod(shape)
    if N != P:
        raise ValueError(f"N={N} does not match prod(shape)={P} for shape={list(shape)}")

    return x.reshape(B, *shape, C)


def flatten_bspatialc(
    x: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    (B,*shape,C) -> (B,N,C)
    """
    if not is_bspatialc(x, shape):
        raise ValueError(f"flatten_bspatialc expects (B,*shape,C) with shape={list(shape)}, got {tuple(x.shape)}")

    B = x.shape[0]
    C = x.shape[-1]
    P = prod(shape)
    return x.reshape(B, P, C)


def unflatten_bsnc(
    x: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    (B,S,N,C) -> (B,S,*shape,C)
    """
    if x.ndim != 4:
        raise ValueError(f"unflatten_bsnc expects (B,S,N,C), got {tuple(x.shape)}")

    B, S, N, C = x.shape
    P = prod(shape)
    if N != P:
        raise ValueError(f"N={N} does not match prod(shape)={P} for shape={list(shape)}")

    return x.reshape(B, S, *shape, C)


def flatten_bsspatialc(
    x: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    """
    (B,S,*shape,C) -> (B,S,N,C)
    """
    if not is_bsspatialc(x, shape):
        raise ValueError(f"flatten_bsspatialc expects (B,S,*shape,C) with shape={list(shape)}, got {tuple(x.shape)}")

    B, S = x.shape[0], x.shape[1]
    C = x.shape[-1]
    P = prod(shape)
    return x.reshape(B, S, P, C)


# -------------------------
# Coords reshape helpers
# -------------------------
def unflatten_coords(
    coords: Optional[torch.Tensor],
    shape: Sequence[int],
    *,
    batch_size: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Unflatten coords consistently with x.

    Supported inputs:
      - (N, d)          -> (*shape, d)
      - (B, N, d)       -> (B, *shape, d)
      - (*shape, d)     -> (*shape, d) (no-op)
      - (B, *shape, d)  -> (B, *shape, d) (no-op)

    If batch_size is provided and coords is (N,d), you can still keep it as (*shape,d).
    If you want (B,*shape,d), pass coords as (B,N,d) before calling.
    """
    if coords is None:
        return None

    P = prod(shape)

    if coords.ndim == 2:
        # (N,d) -> (*shape,d)
        N, d = coords.shape
        if N != P:
            raise ValueError(f"coords N={N} != prod(shape)={P} for shape={list(shape)}")
        return coords.reshape(*shape, d)

    if coords.ndim == 3:
        # (B,N,d) -> (B,*shape,d)
        B, N, d = coords.shape
        if N != P:
            raise ValueError(f"coords N={N} != prod(shape)={P} for shape={list(shape)}")
        if batch_size is not None and B != batch_size:
            raise ValueError(f"coords batch B={B} != batch_size={batch_size}")
        return coords.reshape(B, *shape, d)

    # already spatial?
    if coords.ndim == len(shape) + 1:
        # (*shape,d)
        if list(coords.shape[:-1]) != list(shape):
            raise ValueError(f"coords spatial mismatch: got {tuple(coords.shape[:-1])}, expected {tuple(shape)}")
        return coords

    if coords.ndim == len(shape) + 2:
        # (B,*shape,d)
        if list(coords.shape[1:-1]) != list(shape):
            raise ValueError(f"coords spatial mismatch: got {tuple(coords.shape[1:-1])}, expected {tuple(shape)}")
        if batch_size is not None and coords.shape[0] != batch_size:
            raise ValueError(f"coords batch B={coords.shape[0]} != batch_size={batch_size}")
        return coords

    raise ValueError(f"Unsupported coords shape: {tuple(coords.shape)}")


def flatten_coords(
    coords: Optional[torch.Tensor],
    shape: Sequence[int],
) -> Optional[torch.Tensor]:
    """
    Flatten coords consistently back to N.

    Supported inputs:
      - (*shape, d)     -> (N, d)
      - (B, *shape, d)  -> (B, N, d)
      - (N, d) or (B,N,d) -> no-op
    """
    if coords is None:
        return None

    P = prod(shape)

    if coords.ndim == 2:
        return coords  # (N,d)
    if coords.ndim == 3 and coords.shape[1] == P:
        return coords  # (B,N,d)

    if coords.ndim == len(shape) + 1:
        # (*shape,d) -> (N,d)
        if list(coords.shape[:-1]) != list(shape):
            raise ValueError(f"coords spatial mismatch: got {tuple(coords.shape[:-1])}, expected {tuple(shape)}")
        d = coords.shape[-1]
        return coords.reshape(P, d)

    if coords.ndim == len(shape) + 2:
        # (B,*shape,d) -> (B,N,d)
        if list(coords.shape[1:-1]) != list(shape):
            raise ValueError(f"coords spatial mismatch: got {tuple(coords.shape[1:-1])}, expected {tuple(shape)}")
        B = coords.shape[0]
        d = coords.shape[-1]
        return coords.reshape(B, P, d)

    raise ValueError(f"Unsupported coords shape: {tuple(coords.shape)}")


# -------------------------
# Context API for models
# -------------------------
@dataclass(frozen=True)
class SpatialContext:
    """
    Remembers how we transformed x (and coords) so we can restore outputs.

    - shape: spatial shape used
    - mode: "bnc" | "bsnc" | "spatial" | "sspatial" | "none"
    """
    shape: Optional[List[int]]
    mode: str  # "bnc" / "bsnc" / "spatial" / "sspatial" / "none"

    def restore_x(self, y: torch.Tensor) -> torch.Tensor:
        """Restore model output back to (B,N,C) or (B,S,N,C) if it was unflattened."""
        if self.shape is None or self.mode == "none":
            return y

        if self.mode == "bnc":
            # y is (B,*shape,C) -> (B,N,C)
            return flatten_bspatialc(y, self.shape)

        if self.mode == "bsnc":
            # y is (B,S,*shape,C) -> (B,S,N,C)
            return flatten_bsspatialc(y, self.shape)

        # If input was already spatial, keep output as-is (caller decides)
        return y

    def restore_coords(self, coords: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.shape is None or self.mode == "none":
            return coords
        # If we unflattened coords, this brings it back.
        return flatten_coords(coords, self.shape)


def to_spatial(
    x: torch.Tensor,
    *,
    shape: Optional[Sequence[int]] = None,
    coords: Optional[torch.Tensor] = None,
    geom: Optional[Dict[str, Any]] = None,
    require_shape: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], SpatialContext]:
    """
    Convert x (and optionally coords) to spatial layout if shape is provided/resolved.

    Inputs:
      x:
        - (B,N,C) or (B,S,N,C)  -> unflatten to spatial
        - already spatial (B,*shape,C) or (B,S,*shape,C) -> no-op

    Returns:
      x_spatial, coords_spatial, ctx
      ctx.restore_x(y_spatial) returns back to (B,N,C) or (B,S,N,C)
    """
    shp = resolve_shape(shape, geom)
    if shp is None:
        if require_shape:
            # caller expects spatial conversion but no shape provided
            return x, coords, SpatialContext(shape=None, mode="none")
        return x, coords, SpatialContext(shape=None, mode="none")

    # x: (B,N,C) -> (B,*shape,C)
    if is_bnc(x):
        x2 = unflatten_bnc(x, shp)
        c2 = unflatten_coords(coords, shp, batch_size=x.shape[0]) if coords is not None else None
        return x2, c2, SpatialContext(shape=shp, mode="bnc")

    # x: (B,S,N,C) -> (B,S,*shape,C)
    if is_bsnc(x):
        x2 = unflatten_bsnc(x, shp)
        # coords for rollout is typically static; support (N,d)/(B,N,d)/(spatial) only
        c2 = unflatten_coords(coords, shp, batch_size=None) if coords is not None else None
        return x2, c2, SpatialContext(shape=shp, mode="bsnc")

    # already spatial one-step
    if is_bspatialc(x, shp):
        c2 = unflatten_coords(coords, shp, batch_size=x.shape[0]) if coords is not None else coords
        return x, c2, SpatialContext(shape=shp, mode="spatial")

    # already spatial rollout
    if is_bsspatialc(x, shp):
        c2 = unflatten_coords(coords, shp, batch_size=None) if coords is not None else coords
        return x, c2, SpatialContext(shape=shp, mode="sspatial")

    raise ValueError(f"Unsupported x shape {tuple(x.shape)} for shape={shp}")
