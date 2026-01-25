# datasets/__init__.py
from .ns2d import NS2DDataset
from .ns3d import NS3DDataset
from .carra import CarraDataset
from .airfoil_time import AirfoilTimeDataset


DATASET_REGISTRY = {
    'ns2d': NS2DDataset,
    'ns3d': NS3DDataset,
    'carra': CarraDataset,
    'airfoil_time': AirfoilTimeDataset,
}

__all__ = [
    'DATASET_REGISTRY',
    'NS2DDataset',
    'NS3DDataset',
    'CarraDataset',
    'AirfoilTimeDataset',
]
