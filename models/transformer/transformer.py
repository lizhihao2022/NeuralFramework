# models/transformer/transformer.py
import math
from typing import Optional, Dict, Any

from torch import nn, Tensor

from ..base import (
    MultiHeadSelfAttention,
    BaseMLP,
    RoPE1DAdapter,
    RoPE2DAdapter,
    RoPE3DAdapter,
    unified_pos_embedding
)


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block:

        x -> LN -> MHSA -> Drop -> + (residual)
        x -> LN -> MLP  -> Drop -> + (residual)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_dropout: float = 0.0,
        activation: str = "gelu",
        use_flash: bool = False,
        rope_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_flash=use_flash,
            rope_adapter=rope_adapter,
        )

        # FFN implemented via BaseMLP (no internal residual; residual is in the block)
        self.mlp = BaseMLP(
            in_dim=d_model,
            out_dim=d_model,
            hidden_dims=int(d_model * mlp_ratio),
            activation=activation,
            dropout=mlp_dropout,
            use_residual=False,
            residual_proj=False,
            last_activation=False,
        )

        self.dropout1 = nn.Dropout(proj_dropout)
        self.dropout2 = nn.Dropout(proj_dropout)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        rope_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Args:
            x:          (B, N, d_model)
            attn_bias:  (N, N) or (B, N, N), added to attention logits.
            rope_kwargs: optional kwargs passed to rope_adapter inside attention.
        """
        # Self-attention sub-layer
        x_res = x
        x_norm = self.norm1(x)
        x_attn = self.self_attn(
            x_norm,
            attn_mask=attn_bias,
            key_padding_mask=None,
            rope_kwargs=rope_kwargs,
        )
        x = x_res + self.dropout1(x_attn)

        # MLP sub-layer
        x_res = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x_res + self.dropout2(x_mlp)

        return x


class Transformer(nn.Module):
    """
    Generic Transformer for grid-based PDE fields.

    Input:
        x: (B, *spatial, C_in)
           - spatial can be 1D (L), 2D (H, W), 3D (T, H, W), etc.
    Output:
        y: (B, *spatial, C_out)

    RoPE usage:
        - rope_ndim = 1: 1D RoPE (coords: (B, *spatial, 1))
        - rope_ndim = 2: 2D RoPE (coords: (B, *spatial, 2))
        - rope_ndim = 3: 3D RoPE (coords: (B, *spatial, 3))
    """

    def __init__(
        self,
        model_params: dict,
        **kwargs,
    ) -> None:
        super().__init__()

        # Basic model hyper-parameters
        self.in_channels = model_params.get("in_channels", 1)
        self.out_channels = model_params.get("out_channels", 1)
        self.d_model = model_params.get("d_model", 64)
        self.num_heads = model_params.get("num_heads", 4)
        self.num_layers = model_params.get("num_layers", 4)

        # RoPE control
        self.use_rope = model_params.get("use_rope", False)
        self.rope_ndim = model_params.get("rope_ndim", 2)  # 1, 2, or 3
        
        # Pos embedding option (unified PE)
        self.use_uni_pos = model_params.get("use_uni_pos", False)

        attn_dropout = model_params.get("attn_dropout", 0.0)
        proj_dropout = model_params.get("proj_dropout", 0.0)
        mlp_ratio = model_params.get("mlp_ratio", 4.0)
        mlp_dropout = model_params.get("mlp_dropout", 0.0)
        activation = model_params.get("activation", "gelu")
        use_flash = model_params.get("use_flash", False)

        # Project input features to model dimension
        # in_channels already includes any extra channels (e.g. concatenated coords)
        self.input_proj = nn.Linear(self.in_channels, self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.out_channels)

        # RoPE adapter selection
        if self.use_rope:
            if self.rope_ndim is None:
                raise ValueError("rope_ndim must be set to 1/2/3 when use_rope=True.")
            assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads."
            head_dim = self.d_model // self.num_heads

            if self.rope_ndim == 1:
                self.rope_adapter: Optional[nn.Module] = RoPE1DAdapter(head_dim=head_dim)
            elif self.rope_ndim == 2:
                self.rope_adapter = RoPE2DAdapter(head_dim=head_dim)
            elif self.rope_ndim == 3:
                self.rope_adapter = RoPE3DAdapter(head_dim=head_dim)
            else:
                raise ValueError(f"Unsupported rope_ndim={self.rope_ndim}. Use 1, 2, or 3.")
        else:
            self.rope_adapter = None

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
                activation=activation,
                use_flash=use_flash,
                rope_adapter=self.rope_adapter,
            )
            for _ in range(self.num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:
                Tensor of shape (B, *spatial, C_in), e.g.
                    1D: (B, L, C_in)
                    2D: (B, H, W, C_in)
                    3D: (B, T, H, W, C_in)
                in_channels in config must match C_in.

            coords:
                Optional coordinates used for RoPE.
                Shape: (B, *spatial, rope_ndim), with rope_ndim = 1 / 2 / 3.
                Values are usually normalized to [0, 1].

                If you also want coords as raw input features, concatenate them
                to x before calling forward and increase in_channels accordingly.

            attn_bias:
                Optional attention bias tensor:
                    - shape (N, N) or (B, N, N)
                    - N = product of spatial dimensions
                This is added to attention logits (e.g., from unified_pos_embedding).

        Returns:
            y: Tensor of shape (B, *spatial, C_out)
        """
        assert x.dim() >= 3, "x must be of shape (B, *spatial, C_in)."

        B = x.shape[0]
        spatial_shape = x.shape[1:-1]   # tuple of 1D/2D/3D...
        C_in = x.shape[-1]
        assert C_in == self.in_channels, \
            f"Expected in_channels={self.in_channels}, but got {C_in}."
        
        # Flatten spatial dimensions to token sequence
        N = int(math.prod(spatial_shape))  # number of tokens
        x = x.view(B, N, C_in)             # (B, N, C_in)
        coords = x[..., :self.rope_ndim] if C_in > 1 else None  # (B, N, rope_ndim) or None
        x = self.input_proj(x)             # (B, N, d_model)

        # Build RoPE kwargs if needed
        if self.use_rope:
            if coords is None:
                raise ValueError("coords must be provided when use_rope=True.")

            if coords.dim() != x.dim():  # (B, *spatial, rope_ndim)
                raise ValueError(
                    "coords must have shape (B, *spatial, rope_ndim). "
                    f"Got coords.dim()={coords.dim()}, expected {x.dim()}."
                )

            d_coord = coords.shape[-1]
            if d_coord < self.rope_ndim:
                raise ValueError(
                    f"coords last dim={d_coord} < rope_ndim={self.rope_ndim}."
                )

            coords_flat = coords.view(B, N, d_coord)  # (B, N, d_coord)

            if self.rope_ndim == 1:
                coords_1d = coords_flat[..., 0]  # (B, N)
                rope_kwargs = {"coords": coords_1d}
            elif self.rope_ndim == 2:
                coords_x = coords_flat[..., 0]   # (B, N)
                coords_y = coords_flat[..., 1]   # (B, N)
                rope_kwargs = {"coords_x": coords_x, "coords_y": coords_y}
            elif self.rope_ndim == 3:
                coords_x = coords_flat[..., 0]
                coords_y = coords_flat[..., 1]
                coords_z = coords_flat[..., 2]
                rope_kwargs = {
                    "coords_x": coords_x,
                    "coords_y": coords_y,
                    "coords_z": coords_z,
                }
            else:
                rope_kwargs = None
        else:
            rope_kwargs = None
        
        # Add unified positional embedding if enabled
        if self.use_uni_pos:
            attn_bias = unified_pos_embedding(
                shape_list=spatial_shape,
                ref = min(spatial_shape),
                batch_size=B,
                device=x.device,
            )  # (B, N, N)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x, attn_bias=attn_bias, rope_kwargs=rope_kwargs)

        x = self.norm_out(x)
        x = self.output_proj(x)            # (B, N, C_out)

        # Reshape back to grid
        y = x.view(B, *spatial_shape, self.out_channels)
        return y
