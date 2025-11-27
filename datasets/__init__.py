# datasets/__init__.py
from .ns2d import NS2DDataset


DATASET_REGISTRY = {
    'ns2d': NS2DDataset,
}
