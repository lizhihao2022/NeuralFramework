# models/base/base.py
import math
from typing import Sequence, Optional, Union

import torch
from torch import nn

from .utils import get_activation


class BaseMLP(nn.Module):
    """
    Generic MLP block.

    Features:
      - Flexible hidden sizes
      - Configurable activation
      - Optional dropout between layers
      - Optional residual connection (with optional linear projection)

    Input / output:
      - Accepts tensor of shape (..., in_dim)
      - Returns tensor of shape (..., out_dim)
      - Residual connection is applied on the last dimension

    Args:
        in_dim: input feature dimension
        out_dim: output feature dimension
        hidden_dims: list/tuple of hidden sizes (e.g. [256, 256])
        activation: activation name (e.g. 'gelu', 'relu', 'silu', 'tanh', 'leaky_relu', 'identity')
        dropout: dropout probability applied after each hidden layer
        use_residual: whether to use residual connection
        residual_proj: if True, learn a linear projection when in_dim != out_dim;
                       if False, only add residual when in_dim == out_dim
        last_activation: if True, also apply activation after the last linear layer
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, Sequence[int]]] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_residual: bool = False,
        residual_proj: bool = True,
        last_activation: bool = False,
    ) -> None:
        super().__init__()

        # Normalize hidden_dims to a list
        if hidden_dims is None:
            hidden_dims_list: list[int] = []
        elif isinstance(hidden_dims, int):
            hidden_dims_list = [hidden_dims]
        else:
            hidden_dims_list = list(hidden_dims)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.last_activation = last_activation and (out_dim is not None)

        # Build main MLP stack
        dims = [in_dim] + hidden_dims_list + [out_dim]
        act = get_activation(activation)
        layers = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))

            is_last_layer = (i == len(dims) - 2)
            if not is_last_layer or self.last_activation:
                if activation is not None:
                    layers.append(act.__class__())  # new instance
            if (not is_last_layer) and dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        # Residual projection if needed
        if use_residual:
            if in_dim == out_dim:
                self.residual_proj = nn.Identity()
            elif residual_proj:
                self.residual_proj = nn.Linear(in_dim, out_dim)
            else:
                # Residual requested but dimensions mismatch and projection disabled
                # In this case, we simply skip residual connection.
                self.residual_proj = None
        else:
            self.residual_proj = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Basic parameter initialization for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (..., in_dim)

        Returns:
            Tensor of shape (..., out_dim)
        """
        # Flatten last dimension only, keep leading shape intact
        orig = x
        y = self.net(x)

        if self.use_residual and self.residual_proj is not None:
            res = self.residual_proj(orig)
            y = y + res

        return y
