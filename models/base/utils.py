# models/base/utils.py
from typing import Optional

from torch import nn


def get_activation(name: Optional[str]) -> nn.Module:
    """
    Return an activation module by name.
    If name is None or 'identity', returns nn.Identity().
    """
    if name is None or name.lower() == "identity":
        return nn.Identity()
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
