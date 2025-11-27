# models/__init__.py
from .mlp import MLP

MODEL_REGISTRY = {
    "MLP": MLP,
}
