# trainers/__init__.py
from .base import BaseTrainer

TRAINER_REGISTRY = {
    'MLP': BaseTrainer,
    'Transformer': BaseTrainer,
}

__all__ = ['BaseTrainer', 'TRAINER_REGISTRY']