import torch
import math
from einops import rearrange
from typing import Any, Dict, Tuple, Optional
from .addition import NystromAttention


def Attention_Vanilla(q, k, v):
    score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
    r = torch.einsum("bhij,bhjc->bhic", score, v)
    return r


def Attention_Linear_GNOT(q, k, v):
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-1)
    k_sum = k.sum(dim=-2, keepdim=True)
    inv = 1. / (q * k_sum).sum(dim=-1, keepdim=True)
    r = q + (q @ (k.transpose(-2, -1) @ v)) * inv
    return r


class LinearAttention_Galerkin_and_Fourier(torch.nn.Module):
    def __init__(self, attn_type, n_dim, n_head, use_ln=False):
        super().__init__()
        self.attn_type = attn_type
        self.n_dim = n_dim
        self.n_head = n_head
        self.dim_head = self.n_dim // self.n_head
        self.use_ln = use_ln
        self.to_qkv = torch.nn.Linear(n_dim, n_dim*3, bias = False)
        self.project_out = (not self.n_head == 1)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.v_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.k_norm = torch.nn.LayerNorm(self.dim_head)
                self.v_norm = torch.nn.LayerNorm(self.dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = torch.nn.InstanceNorm1d(self.dim_head)
                self.k_norm = torch.nn.InstanceNorm1d(self.dim_head)
            else:
                self.q_norm = torch.nn.LayerNorm(self.dim_head)
                self.k_norm = torch.nn.LayerNorm(self.dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim),
            torch.nn.Dropout(0.0)
        ) if self.project_out else torch.nn.Identity()

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), qkv)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        elif self.attn_type == "fourier":
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)
        else:
            raise NotImplementedError("Invalid Attention Type!")

        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


ACTIVATION = {"Sigmoid": torch.nn.Sigmoid(),
              "Tanh": torch.nn.Tanh(),
              "ReLU": torch.nn.ReLU(),
              "LeakyReLU": torch.nn.LeakyReLU(0.1),
              "ELU": torch.nn.ELU(),
              "GELU": torch.nn.GELU()
              }


ATTENTION = {"Attention_Vanilla": Attention_Vanilla,
             "Attention_Linear_GNOT": Attention_Linear_GNOT
            }


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.act = act
        self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)])
        self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        r = self.act(self.input(x))
        for i in range(0, self.n_layer):
            r = r + self.act(self.hidden[i](r))
        r = self.output(r)
        return r
        

class LNO(torch.nn.Module):
    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.attn = attn
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)
        
        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            r = self.attn(q, k, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r
    
    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, attn, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.attn = attn
            self.act = act
            
            if self.attn == "Galerkin":
                self.self_attn = LinearAttention_Galerkin_and_Fourier('galerkin', self.n_dim, self.n_head, use_ln=True)
            elif self.attn == "Nystrom":
                self.self_attn = NystromAttention(self.n_dim, heads =self.n_head, dim_head=self.n_dim//self.n_head, dropout=0.0)
            else:
                self.self_attn = LNO.SelfAttention(self.n_mode, self.n_dim, self.n_head, ATTENTION[self.attn])
            
            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)
            
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim*2),
                self.act,
                torch.nn.Linear(self.n_dim*2, self.n_dim),
            )

        def forward(self, y):   
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y

        
    def __init__(self, model_params):
                #  n_block, n_mode, n_dim, n_head, n_layer, x_dim, y1_dim, y2_dim, attn, act, model_attr):
        super().__init__()
        
        self.n_block = model_params["n_block"]
        self.n_mode = model_params["n_mode"]
        self.n_dim = model_params["n_dim"]
        self.n_head = model_params["n_head"]
        self.n_layer = model_params["n_layer"]
        self.act = ACTIVATION[model_params["act"]]
        self.attn = model_params["attn"]
        
        self.x_dim = model_params["x_dim"]
        self.y1_dim = model_params["y_dim"] + model_params['coord_dim']
        if model_params["time"]:
            self.y2_dim = 1
        else:
            self.y2_dim = model_params["y_dim"]
        
        self.trunk_projector = MLP(self.x_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = MLP(self.y1_dim, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = MLP(self.n_dim, self.n_dim, self.y2_dim, self.n_layer, self.act)
        self.attention_projector = MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(*[LNO.AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.attn, self.act) for _ in range(0, self.n_block)])
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        geom: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
        ) -> torch.Tensor:
        if coords is not None:
            y = torch.concatenate([x, coords], dim=-1)
            y = self.branch_projector(y)
        else:
            y = self.branch_projector(x)
        x = self.trunk_projector(coords)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)
        
        z = torch.einsum("bij,bic->bjc", score_encode, y)
        
        for block in self.attn_blocks:
            z = block(z)
        
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        return r
