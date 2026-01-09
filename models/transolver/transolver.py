# models/transolver/transolver.py
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from models.base import timestep_embedding, unified_pos_embedding
from models.transolver.basic import MLP
from models.transolver.physics_attention import PhysicsAttentionIrregularMesh
from models.transolver.physics_attention import PhysicsAttentionStructuredMesh1D
from models.transolver.physics_attention import PhysicsAttentionStructuredMesh2D
from models.transolver.physics_attention import PhysicsAttentionStructuredMesh3D

PHYSICS_ATTENTION = {
    'unstructured': PhysicsAttentionIrregularMesh,
    'structured_1D': PhysicsAttentionStructuredMesh1D,
    'structured_2D': PhysicsAttentionStructuredMesh2D,
    'structured_3D': PhysicsAttentionStructuredMesh3D
}


class TransolverBlock(nn.Module):
    """Transolver encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            geotype='unstructured',
            shapelist=None
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = PHYSICS_ATTENTION[geotype](hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                               dropout=dropout, slice_num=slice_num, shapelist=shapelist)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(nn.Module):
    def __init__(self, model_params):
        super(Transolver, self).__init__()
        self.__name__ = 'Transolver'
        for key, value in model_params.items():
            setattr(self, key, value)
        ## embedding
        if self.unified_pos and self.geotype != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(self.shapelist, self.ref)
            self.preprocess = MLP(self.fun_dim + self.ref ** len(self.shapelist), self.n_hidden * 2,
                                  self.n_hidden, n_layers=0, res=False, act=self.act)
        else:
            self.preprocess = MLP(self.fun_dim + self.space_dim, self.n_hidden * 2, self.n_hidden,
                                  n_layers=0, res=False, act=self.act)
        if self.time_input:
            self.time_fc = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.SiLU(),
                                         nn.Linear(self.n_hidden, self.n_hidden))

        ## models
        self.blocks = nn.ModuleList([TransolverBlock(num_heads=self.n_heads, hidden_dim=self.n_hidden,
                                                      dropout=self.dropout,
                                                      act=self.act,
                                                      mlp_ratio=self.mlp_ratio,
                                                      out_dim=self.out_dim,
                                                      slice_num=self.slice_num,
                                                      last_layer=(_ == self.n_layers - 1),
                                                      geotype=self.geotype,
                                                      shapelist=self.shapelist)
                                     for _ in range(self.n_layers)])
        self.placeholder = nn.Parameter((1 / (self.n_hidden)) * torch.rand(self.n_hidden, dtype=torch.float))
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

    def structured_geo(self, x, fx, T=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def unstructured_geo(self, x, fx=None, T=None):
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)
        return fx

    def forward(
        self,
        x: torch.Tensor,
        fx=None,
        T=None,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.geotype == 'unstructured':
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)
