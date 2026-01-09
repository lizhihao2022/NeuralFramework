# models/lsm/lsm.py
from typing import Any, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import MLP
from ..base import timestep_embedding, unified_pos_embedding
from .neural_spectral_block import NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D
from .unet_block import DoubleConv1D, Down1D, Up1D, OutConv1D, DoubleConv2D, Down2D, Up2D, OutConv2D, \
    DoubleConv3D, Down3D, Up3D, OutConv3D
from .geo_projection import SpectralConv2d_IrregularGeo, IPHI

ConvList = [None, DoubleConv1D, DoubleConv2D, DoubleConv3D]
DownList = [None, Down1D, Down2D, Down3D]
UpList = [None, Up1D, Up2D, Up3D]
OutList = [None, OutConv1D, OutConv2D, OutConv3D]
BlockList = [None, NeuralSpectralBlock1D, NeuralSpectralBlock2D, NeuralSpectralBlock3D]


class LSM(nn.Module):
    def __init__(self, model_params: dict):
        super(LSM, self).__init__()
        self.__name__ = 'LSM'
        
        for key, value in model_params.items():
            setattr(self, key, value)
        
        if model_params['task'] == 'steady':
            normtype = 'bn'
        else:
            normtype = 'in' # when conducting dynamic tasks, use instance norm for stability
        ## embedding
        if model_params['unified_pos'] and model_params['geotype'] != 'unstructured':  # only for structured mesh
            self.pos = unified_pos_embedding(model_params['shapelist'], model_params['ref'])
            self.preprocess = MLP(model_params['fun_dim'] + model_params['ref'] ** len(model_params['shapelist']), model_params['n_hidden'] * 2,
                                  model_params['n_hidden'], n_layers=0, res=False, act=model_params['act'])
        else:
            self.preprocess = MLP(model_params['fun_dim'] + model_params['space_dim'], model_params['n_hidden'] * 2, model_params['n_hidden'],
                                  n_layers=0, res=False, act=self.act)
        if self.time_input:
            self.time_fc = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.SiLU(),
                                         nn.Linear(self.n_hidden, self.n_hidden))
        # geometry projection
        if self.geotype == 'unstructured':
            self.fftproject_in = SpectralConv2d_IrregularGeo(self.n_hidden, self.n_hidden, self.modes, self.modes,
                                                             self.s1, self.s2)
            self.fftproject_out = SpectralConv2d_IrregularGeo(self.n_hidden, self.n_hidden, self.modes, self.modes,
                                                              self.s1, self.s2)
            self.iphi = IPHI()
            patch_size = [(size + (16 - size % 16) % 16) // 16 for size in [self.s1, self.s2]]
            self.padding = [(16 - size % 16) % 16 for size in [self.s1, self.s2]]
        else:
            patch_size = [(size + (16 - size % 16) % 16) // 16 for size in self.shapelist]
            self.padding = [(16 - size % 16) % 16 for size in self.shapelist]
        # multiscale modules
        self.inc = ConvList[len(patch_size)](self.n_hidden, self.n_hidden, normtype=normtype)
        self.down1 = DownList[len(patch_size)](self.n_hidden, self.n_hidden * 2, normtype=normtype)
        self.down2 = DownList[len(patch_size)](self.n_hidden * 2, self.n_hidden * 4, normtype=normtype)
        self.down3 = DownList[len(patch_size)](self.n_hidden * 4, self.n_hidden * 8, normtype=normtype)
        factor = 2 if self.bilinear else 1
        self.down4 = DownList[len(patch_size)](self.n_hidden * 8, self.n_hidden * 16 // factor, normtype=normtype)
        self.up1 = UpList[len(patch_size)](self.n_hidden * 16, self.n_hidden * 8 // factor, self.bilinear, normtype=normtype)
        self.up2 = UpList[len(patch_size)](self.n_hidden * 8, self.n_hidden * 4 // factor, self.bilinear, normtype=normtype)
        self.up3 = UpList[len(patch_size)](self.n_hidden * 4, self.n_hidden * 2 // factor, self.bilinear, normtype=normtype)
        self.up4 = UpList[len(patch_size)](self.n_hidden * 2, self.n_hidden, self.bilinear, normtype=normtype)
        self.outc = OutList[len(patch_size)](self.n_hidden, self.n_hidden)
        # Patchified Neural Spectral Blocks
        self.process1 = BlockList[len(patch_size)](self.n_hidden, self.num_basis, patch_size, self.num_token, self.n_heads)
        self.process2 = BlockList[len(patch_size)](self.n_hidden * 2, self.num_basis, patch_size, self.num_token, self.n_heads)
        self.process3 = BlockList[len(patch_size)](self.n_hidden * 4, self.num_basis, patch_size, self.num_token, self.n_heads)
        self.process4 = BlockList[len(patch_size)](self.n_hidden * 8, self.num_basis, patch_size, self.num_token, self.n_heads)
        self.process5 = BlockList[len(patch_size)](self.n_hidden * 16 // factor, self.num_basis, patch_size, self.num_token,
                                                   self.n_heads)
        # projectors
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden * 2)
        self.fc2 = nn.Linear(self.n_hidden * 2, self.out_dim)

    def structured_geo(self, x, fx, T=None):
        B, N, _ = x.shape
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        x = fx.permute(0, 2, 1).reshape(B, self.n_hidden, *self.shapelist)
        if not all(item == 0 for item in self.padding):
            if len(self.shapelist) == 2:
                x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
            elif len(self.shapelist) == 3:
                x = F.pad(x, [0, self.padding[2], 0, self.padding[1], 0, self.padding[0]])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            if len(self.shapelist) == 2:
                x = x[..., :-self.padding[0], :-self.padding[1]]
            elif len(self.shapelist) == 3:
                x = x[..., :-self.padding[0], :-self.padding[1], :-self.padding[2]]
        x = x.reshape(B, self.n_hidden, -1).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def unstructured_geo(self, x, fx=None, T=None):
        original_pos = x
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        x = self.fftproject_in(fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)
        x = self.fftproject_out(x, x_out=original_pos, iphi=self.iphi, code=None).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

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
