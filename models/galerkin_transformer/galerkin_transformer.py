import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from .basic import MLP, LinearAttention

import os.path as osp


class GalerkinTransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
    ):
        super().__init__()
        self.__file__ = osp.abspath(__file__)
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1a = nn.LayerNorm(hidden_dim)
        self.Attn = LinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                    dropout=dropout, attn_type='galerkin')
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx), self.ln_1a(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class GalerkinTransformer(nn.Module):
    def __init__(self, 
                 model_params: dict, 
                 **kwargs
                 ) -> None:
        super(GalerkinTransformer, self).__init__()
        self.__file__ = osp.abspath(__file__)
        self.__name__ = 'GalerkinTransformer'
        
        fun_dim = model_params.get('fun_dim', 1)
        space_dim = model_params.get('space_dim', 1)
        n_hidden = model_params.get('n_hidden', 128)
        out_dim = model_params.get('out_dim', 1)
        n_layers = model_params.get('n_layers', 6)
        n_heads = model_params.get('n_heads', 8)
        mlp_ratio = model_params.get('mlp_ratio', 4)
        act = model_params.get('act', 'gelu')
        dropout = model_params.get('dropout', 0.1)
        
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        ## models
        self.blocks = nn.ModuleList([GalerkinTransformerBlock(num_heads=n_heads, hidden_dim=n_hidden,
                                                                dropout=dropout,
                                                                act=act,
                                                                mlp_ratio=mlp_ratio,
                                                                out_dim=out_dim,
                                                                last_layer=(_ == n_layers - 1)) for _ in range(n_layers)])
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, H, W, C = x.size()
        x = x.view(B, H * W, C)
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
            
        fx = fx.view(B, H, W, -1)
        return fx
