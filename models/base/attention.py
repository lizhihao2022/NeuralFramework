# models/base/attention.py
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .embedding import RotaryEmbedding1D, rotary_pos_embedding


# ============================================================
# Rotary Positional Embedding (RoPE)
# ============================================================
class RoPE1DAdapter(nn.Module):
    """
    1D RoPE adapter for a single coordinate axis (e.g. x or t).

    Assumes head_dim is even:
        q, k: (B, num_heads, N, head_dim)
        coords: (B, N)
    """

    def __init__(self, head_dim: int, base: float = 10000.0, scale: float = 1.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for 1D RoPE."

        self.head_dim = head_dim
        self.rope = RotaryEmbedding1D(head_dim, base=base, scale=scale)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        coords: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q, k:   (B, num_heads, N, head_dim)
            coords: (B, N)     1D coordinates in [0, 1] or physical range

        Returns:
            q_out, k_out: same shapes as q, k
        """
        # coords embedding: (B, N, head_dim)
        cos, sin = self.rope(coords)  # (B, N, head_dim)

        # broadcast over heads
        cos = cos.unsqueeze(1)  # (B, 1, N, head_dim)
        sin = sin.unsqueeze(1)  # (B, 1, N, head_dim)

        q_out = rotary_pos_embedding(q, cos, sin)
        k_out = rotary_pos_embedding(k, cos, sin)
        return q_out, k_out


class RoPE2DAdapter(nn.Module):
    """
    2D RoPE adapter for (x, y) coordinates.

    Assumes head_dim is split into two equal parts: D = 2 * D_axis, and D_axis is even.
    q, k: (B, num_heads, N, head_dim)
    coords_x, coords_y: (B, N)
    """

    def __init__(self, head_dim: int, base: float = 10000.0, scale: float = 1.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be divisible by 2 for 2D RoPE."
        d_axis = head_dim // 2
        assert d_axis % 2 == 0, "Each axis dim must be even."

        self.d_axis = d_axis
        self.rope_x = RotaryEmbedding1D(d_axis, base=base, scale=scale)
        self.rope_y = RotaryEmbedding1D(d_axis, base=base, scale=scale)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        coords_x: Tensor,
        coords_y: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        q, k: (B, num_heads, N, head_dim)
        coords_x, coords_y: (B, N)
        """
        B, H, N, D = q.shape
        D_axis = self.d_axis

        qx, qy = q[..., :D_axis], q[..., D_axis:]
        kx, ky = k[..., :D_axis], k[..., D_axis:]

        cos_x, sin_x = self.rope_x(coords_x)  # (B, N, D_axis)
        cos_y, sin_y = self.rope_y(coords_y)  # (B, N, D_axis)

        cos_x = cos_x.unsqueeze(1)  # (B, 1, N, D_axis)
        sin_x = sin_x.unsqueeze(1)
        cos_y = cos_y.unsqueeze(1)
        sin_y = sin_y.unsqueeze(1)

        qx = rotary_pos_embedding(qx, cos_x, sin_x)
        kx = rotary_pos_embedding(kx, cos_x, sin_x)
        qy = rotary_pos_embedding(qy, cos_y, sin_y)
        ky = rotary_pos_embedding(ky, cos_y, sin_y)

        q_out = torch.cat([qx, qy], dim=-1)
        k_out = torch.cat([kx, ky], dim=-1)
        return q_out, k_out


class RoPE3DAdapter(nn.Module):
    """
    3D RoPE adapter for (x, y, z) coordinates.

    Assumes head_dim is split into 3 equal parts: D = 3 * D_axis, and D_axis is even.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, scale: float = 1.0) -> None:
        super().__init__()
        assert head_dim % 3 == 0, "head_dim must be divisible by 3 for 3D RoPE."
        d_axis = head_dim // 3
        assert d_axis % 2 == 0, "Each axis dim must be even."

        self.d_axis = d_axis
        self.rope_x = RotaryEmbedding1D(d_axis, base=base, scale=scale)
        self.rope_y = RotaryEmbedding1D(d_axis, base=base, scale=scale)
        self.rope_z = RotaryEmbedding1D(d_axis, base=base, scale=scale)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        coords_x: Tensor,
        coords_y: Tensor,
        coords_z: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        q, k: (B, num_heads, N, head_dim)
        coords_{x,y,z}: (B, N)
        """
        B, H, N, D = q.shape
        D_axis = self.d_axis

        qx, qy, qz = q.split(D_axis, dim=-1)
        kx, ky, kz = k.split(D_axis, dim=-1)

        cos_x, sin_x = self.rope_x(coords_x)  # (B, N, D_axis)
        cos_y, sin_y = self.rope_y(coords_y)
        cos_z, sin_z = self.rope_z(coords_z)

        cos_x = cos_x.unsqueeze(1)
        sin_x = sin_x.unsqueeze(1)
        cos_y = cos_y.unsqueeze(1)
        sin_y = sin_y.unsqueeze(1)
        cos_z = cos_z.unsqueeze(1)
        sin_z = sin_z.unsqueeze(1)

        qx = rotary_pos_embedding(qx, cos_x, sin_x)
        kx = rotary_pos_embedding(kx, cos_x, sin_x)
        qy = rotary_pos_embedding(qy, cos_y, sin_y)
        ky = rotary_pos_embedding(ky, cos_y, sin_y)
        qz = rotary_pos_embedding(qz, cos_z, sin_z)
        kz = rotary_pos_embedding(kz, cos_z, sin_z)

        q_out = torch.cat([qx, qy, qz], dim=-1)
        k_out = torch.cat([kx, ky, kz], dim=-1)
        return q_out, k_out


# ============================================================
# Core Multi-Head Attention (Self / Cross, with optional Flash)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention with optional FlashAttention and RoPE.

    x: (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_flash: bool = False,
        rope_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout_p = attn_dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.use_flash = use_flash
        self.rope_adapter = rope_adapter  # e.g. RoPE2DAdapter / RoPE3DAdapter

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        rope_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, N, d_model)
            attn_mask: optional (N, N) or (B, N, N), additive or bool mask
            key_padding_mask: optional (B, N) bool mask
            rope_kwargs: extra kwargs passed to rope_adapter, e.g.
                {"coords_x": ..., "coords_y": ...}

        Returns:
            out: (B, N, d_model)
        """
        B, N, _ = x.shape

        # Project to q, k, v
        qkv = self.qkv_proj(x)  # (B, N, 3*d_model)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # Optional RoPE
        if self.rope_adapter is not None:
            rope_kwargs = rope_kwargs or {}
            q, k = self.rope_adapter(q, k, **rope_kwargs)

        # FlashAttention path (only if no masks for simplicity)
        if (
            self.use_flash
            and hasattr(F, "scaled_dot_product_attention")
            and attn_mask is None
            and key_padding_mask is None
        ):
            # F.scaled_dot_product_attention expects (B*num_heads, N, head_dim)
            q_ = q
            k_ = k
            v_ = v
            # (B, num_heads, N, head_dim) -> (B*num_heads, N, head_dim)
            q_ = q_.reshape(B * self.num_heads, N, self.head_dim)
            k_ = k_.reshape(B * self.num_heads, N, self.head_dim)
            v_ = v_.reshape(B * self.num_heads, N, self.head_dim)

            attn_out = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )  # (B*num_heads, N, head_dim)

            attn_out = attn_out.view(B, self.num_heads, N, self.head_dim)
        else:
            # Manual attention
            # (B, num_heads, N, head_dim) -> (B, num_heads, N, N)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # key_padding_mask: (B, N) -> (B, 1, 1, N) broadcast
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None, None, :]  # True = pad
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))

            # attn_mask: (N, N) or (B, N, N)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_scores = attn_scores + attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    attn_scores = attn_scores + attn_mask[:, None, :, :]

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, N, head_dim)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, N, num_heads, head_dim)
        attn_out = attn_out.view(B, N, self.d_model)      # (B, N, d_model)
        out = self.out_proj(attn_out)
        out = self.proj_dropout(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention:
        Q来自 x，K/V 来自 context。

    x:       (B, N_q, d_model)
    context: (B, N_kv, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_flash: bool = False,
        rope_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout_p = attn_dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.use_flash = use_flash
        self.rope_adapter = rope_adapter

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        rope_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, N_q, d_model)      - queries
            context: (B, N_kv, d_model) - keys/values
        """
        B, N_q, _ = x.shape
        Bc, N_kv, _ = context.shape
        assert B == Bc, "Batch size mismatch between x and context."

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N_q,Dh)
        k = k.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B,H,N_kv,Dh)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B,H,N_kv,Dh)

        if self.rope_adapter is not None:
            rope_kwargs = rope_kwargs or {}
            q, k = self.rope_adapter(q, k, **rope_kwargs)

        if (
            self.use_flash
            and hasattr(F, "scaled_dot_product_attention")
            and attn_mask is None
            and key_padding_mask is None
        ):
            q_ = q.reshape(B * self.num_heads, N_q, self.head_dim)
            k_ = k.reshape(B * self.num_heads, N_kv, self.head_dim)
            v_ = v.reshape(B * self.num_heads, N_kv, self.head_dim)

            attn_out = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=False,
            )  # (B*H, N_q, Dh)

            attn_out = attn_out.view(B, self.num_heads, N_q, self.head_dim)
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,N_q,N_kv)

            if key_padding_mask is not None:
                # key_padding_mask: (B, N_kv)
                mask = key_padding_mask[:, None, None, :]  # True = pad
                attn_scores = attn_scores.masked_fill(mask, float("-inf"))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_scores = attn_scores + attn_mask[None, None, :, :]
                elif attn_mask.dim() == 3:
                    attn_scores = attn_scores + attn_mask[:, None, :, :]

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            attn_out = torch.matmul(attn_weights, v)  # (B,H,N_q,Dh)

        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, N_q, H, Dh)
        attn_out = attn_out.view(B, N_q, self.d_model)
        out = self.out_proj(attn_out)
        out = self.proj_dropout(out)
        return out


# ============================================================
# Channel Attention (SE-style)
# ============================================================

class ChannelAttention(nn.Module):
    """
    Simple channel attention (Squeeze-and-Excitation style).

    x: (B, N, d_model)
    Global average pool over N, then MLP to get channel-wise gates.
    """

    def __init__(self, d_model: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(d_model // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # (B, C)
        pooled = x.mean(dim=1)
        weights = self.mlp(pooled)  # (B, C)
        return x * weights.unsqueeze(1)


# ============================================================
# Window Attention 2D (Swin-style)
# ============================================================

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """
    x: (B, H, W, C)
    returns: (B * num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, \
        "H and W must be divisible by window_size."

    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C,
    )  # (B, nH, Ws, nW, Ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(
    windows: Tensor,
    window_size: int,
    H: int,
    W: int,
) -> Tensor:
    """
    windows: (B * num_windows, window_size*window_size, C)
    returns: (B, H, W, C)
    """
    B_ = windows.shape[0] // (H // window_size * W // window_size)
    C = windows.shape[-1]
    x = windows.view(
        B_,
        H // window_size, W // window_size,
        window_size, window_size, C,
    )  # (B_, nH, nW, Ws, Ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B_, H, W, C)
    return x


class WindowAttention2D(nn.Module):
    """
    2D windowed self-attention.

    x: (B, N, d_model), where N = H * W
    You must pass H, W at forward time.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_flash: bool = False,
        rope_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_flash=use_flash,
            rope_adapter=rope_adapter,
        )

    def forward(
        self,
        x: Tensor,
        H: int,
        W: int,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        rope_kwargs: Optional[dict] = None,
    ) -> Tensor:
        """
        x: (B, N, d_model), N = H * W
        """
        B, N, C = x.shape
        assert N == H * W, "N must equal H * W."

        x = x.view(B, H, W, C)
        windows = window_partition(x, self.window_size)  # (B*nW, Ws*Ws, C)
        windows = self.attn(
            windows,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rope_kwargs=rope_kwargs,
        )  # (B*nW, Ws*Ws, C)
        x = window_reverse(windows, self.window_size, H, W)  # (B, H, W, C)
        x = x.view(B, N, C)
        return x


# ============================================================
# Axial Attention 2D
# ============================================================

class AxialAttention2D(nn.Module):
    """
    Axial attention on 2D grid:
        - attention along H axis
        - then attention along W axis

    x: (B, N, d_model), N = H * W
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_height = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_flash=use_flash,
        )
        self.attn_width = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_flash=use_flash,
        )

    def forward(
        self,
        x: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        """
        x: (B, N, d_model), N = H * W
        """
        B, N, C = x.shape
        assert N == H * W, "N must equal H * W for AxialAttention2D."

        x = x.view(B, H, W, C)

        # Attention along height (H)
        # reshape to (B*W, H, C)
        x_h = x.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        x_h = self.attn_height(x_h)  # (B*W, H, C)
        x_h = x_h.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()  # (B, H, W, C)

        # Attention along width (W)
        x_w = x_h.view(B * H, W, C)
        x_w = self.attn_width(x_w)  # (B*H, W, C)
        x_w = x_w.view(B, H, W, C)

        out = x_w.view(B, N, C)
        return out
