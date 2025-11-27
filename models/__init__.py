# models/__init__.py
from .mlp import MLP
from .transformer import Transformer

MODEL_REGISTRY = {
    "MLP": MLP,
    "Transformer": Transformer,
}

__all__ = ["MODEL_REGISTRY"]
