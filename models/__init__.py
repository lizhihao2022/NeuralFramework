# models/__init__.py
from .mlp import MLP
from .transformer import Transformer
from .m2no import M2NO2d
from .swin_transformer import SwinTransformerV2, SwinMLP


MODEL_REGISTRY = {
    "MLP": MLP,
    "M2NO2d": M2NO2d,
    "Transformer": Transformer,
    "SwinTransformerV2": SwinTransformerV2,
    "SwinMLP": SwinMLP,
}

__all__ = [
    "MODEL_REGISTRY", 
    "MLP", "Transformer", "M2NO2d", "SwinTransformerV2", "SwinMLP"
    ]
