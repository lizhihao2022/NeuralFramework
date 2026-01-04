import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU
}


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, n_layers: int = 1,
                 act: str = 'gelu', res: bool = True) -> None:
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act_fn = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
    
    
class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0., attn_type: str = 'l1', **kwargs: Optional[dict]) -> None:
        super(LinearAttention, self).__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y is None:
            y = x
        else:
            y = y
        B, T1, _ = x.size()
        _, T2, _ = y.size()
        q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)
        k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)
        v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)

        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1. / T2
        elif self.attn_type == "l2":
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        out = self.attn_drop((q @ context) * D_inv + q)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
