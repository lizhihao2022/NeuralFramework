# models/__init__.py
from .mlp import MLP
from .transformer import Transformer
from .m2no import M2NO2d


MODEL_REGISTRY = {
    "MLP": MLP,
    "Transformer": Transformer,
    "M2NO2d": M2NO2d,
}

__all__ = ["MODEL_REGISTRY"]
