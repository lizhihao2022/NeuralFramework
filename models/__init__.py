# models/__init__.py
from .mlp import MLP
from .transformer import Transformer
from .m2no import M2NO2d
from .swin_transformer import SwinTransformer, SwinTransformerV2, SwinMLP


MODEL_REGISTRY = {
    "MLP": MLP,
    "M2NO2d": M2NO2d,
    "Transformer": Transformer,
    "SwinTransformer": SwinTransformer,
    "SwinTransformerV2": SwinTransformerV2,
    "SwinMLP": SwinMLP,
}

__all__ = [
    "MODEL_REGISTRY", 
    "MLP", "Transformer", "M2NO2d", "SwinTransformer", "SwinTransformerV2", "SwinMLP"
    ]
