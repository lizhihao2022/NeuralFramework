# trainers/__init__.py
from .base import BaseTrainer

TRAINER_REGISTRY = {
    'MLP': BaseTrainer,
    'UNet1d': BaseTrainer,
    'UNet2d': BaseTrainer,
    'UNet3d': BaseTrainer,
    'FNO1d': BaseTrainer,
    'FNO2d': BaseTrainer,
    'FNO3d': BaseTrainer,
    'Transformer': BaseTrainer,
    'M2NO2d': BaseTrainer,
    'SwinTransformerV2': BaseTrainer,
    'SwinMLP': BaseTrainer,
}

__all__ = ['BaseTrainer', 'TRAINER_REGISTRY']
