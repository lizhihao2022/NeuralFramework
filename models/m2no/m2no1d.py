# models/m2no/m2no1d.py
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn

from .grid_operator import LPFOperator1d


class GridBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_ite, bias=True, padding_mode='zeros'):
        super(GridBlock1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ite = num_ite
        
        self.S = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode) for _ in range(num_ite)])
        self.A = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, padding_mode=padding_mode) for _ in range(num_ite)])
    
    def forward(self, f, u=None):
        if u is None:
            u = torch.zeros_like(f).to(f.device)
            
        for i in range(self.num_ite):
            r = f - self.A[i](u)
            u = u + self.S[i](r)

        return r, u


class MultiGrid1d(nn.Module):
    def __init__(self, in_channels, out_channels, grid_levels, op, 
                 bias=True, padding_mode='zeros', resolution=1024, 
                 norm=False, **kwargs):
        super(MultiGrid1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_levels = grid_levels
        self.num_level = len(grid_levels)
        self.norm = norm
        self.op = op
        self.resolution = resolution
        
        self.pre_S = GridBlock1d(in_channels, out_channels, 1, bias, padding_mode)
        self.post_S = GridBlock1d(in_channels, out_channels, 1, bias, padding_mode)
        self.grid_list = nn.ModuleList([GridBlock1d(in_channels, out_channels, grid_levels[i], bias, padding_mode) for i in range(self.num_level)])
        if norm:
            self.norm_list = nn.ModuleList([nn.LayerNorm([out_channels, resolution//2**i]) for i in range(self.num_level)])
    
    def forward(self, f):
        input_resolution = f.shape[2]
        
        u_list = [None] * (self.num_level + 1)
        f_list = [None] * (self.num_level + 1)
        
        r_n, u_n = self.pre_S(f)
        
        while u_n.shape[2] > self.resolution:
            r_n = self.op.restrict(r_n)
            u_n = self.op.restrict(u_n)
        
        while u_n.shape[2] < self.resolution:
            r_n = self.op.prolongate(r_n)
            u_n = self.op.prolongate(u_n)
        
        f_list[0] = r_n
        u_list[0] = u_n
        
        for i in range(self.num_level):
            r = self.op.restrict(f_list[i])
            e = self.op.restrict(u_list[i])
            r, e = self.grid_list[i](r, e)
            f_list[i+1] = r
            u_list[i+1] = e
    
        for i in range(self.num_level, 0, -1):
            if self.norm:
                u_list[i-1] = self.norm_list[i-1](u_list[i-1] + self.op.prolongate(u_list[i]))
            else:
                u_list[i-1] = u_list[i-1] + self.op.prolongate(u_list[i])
        
        u_n = u_list[0]
        
        while u_n.shape[2] > input_resolution:
            u_n = self.op.restrict(u_n)
        
        while u_n.shape[2] < input_resolution:
            u_n = self.op.prolongate(u_n)
        
        r_n, u_n = self.post_S(f, u_n)
                    
        return u_n


class M2NO1d(nn.Module):
    def __init__(self, in_channels, out_channels, c, k, num_layer, 
                 grid_levels, base='legendre', bias=True, resolution=1024, 
                 operator='wavelet', padding_mode='zeros', initializer=None, 
                 norm=True, activation='gelu', **kwargs):
        super(M2NO1d, self).__init__()
        self.num_layer = num_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_channels = c * k

        self.L_in = nn.Linear(in_channels, hidden_channels)
        self.L_hidden = nn.Linear(hidden_channels, hidden_channels)
        self.L_out = nn.Linear(hidden_channels, out_channels)
        
        self.wavelet_operator = LPFOperator1d(c=c, k=k, base=base, bias=bias, padding_mode=padding_mode)
        
        self.conv_list = nn.ModuleList([MultiGrid1d(hidden_channels, hidden_channels, 
                                                    grid_levels, self.wavelet_operator, 
                                                    norm=norm, bias=bias, padding_mode=padding_mode, 
                                                    resolution=resolution) for _ in range(num_layer)])
        
        if activation == 'gelu':
            self.activate = nn.GELU()
        elif activation == 'relu':
            self.activate = nn.ReLU()
        elif activation == 'tanh':
            self.activate = nn.Tanh()
        else:
            raise ValueError('Activation not supported')
        
        if initializer is not None:
            self.reset_parameters(initializer)
    
    def reset_parameters(self, initializer):
        initializer(self.L_in.weight)
        initializer(self.L_hidden.weight)
        initializer(self.L_out.weight)
        
    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        u = self.L_in(x)
        u = u.permute(0, 2, 1)
        for i in range(self.num_layer):
            u = self.conv_list[i](u)
            if i != self.num_layer-1:
                u = self.activate(u)
        u = u.permute(0, 2, 1)
        u = self.activate(self.L_hidden(u))
        u = self.L_out(u)
        
        return u
