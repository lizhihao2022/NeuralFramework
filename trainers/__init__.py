# trainers/__init__.py
from .base import BaseTrainer

TRAINER_REGISTRY = {
    'MLP': BaseTrainer,
    'Transformer': BaseTrainer,
    'M2NO2d': BaseTrainer,
    'SwinTransformerV2': BaseTrainer,
    'SwinMLP': BaseTrainer,
}

__all__ = ['BaseTrainer', 'TRAINER_REGISTRY']
