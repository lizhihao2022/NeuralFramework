# models/__init__.py
from .mlp import MLP
from .unet import UNet1d, UNet2d, UNet3d
from .transformer import Transformer
from .m2no import M2NO2d
from .swin_transformer import SwinTransformerV2, SwinMLP
from .fno import FNO1d, FNO2d, FNO3d


MODEL_REGISTRY = {
    "MLP": MLP,
    "UNet1d": UNet1d,
    "UNet2d": UNet2d,
    "UNet3d": UNet3d,
    "M2NO2d": M2NO2d,
    "FNO1d": FNO1d,
    "FNO2d": FNO2d,
    "FNO3d": FNO3d,
    "Transformer": Transformer,
    "SwinTransformerV2": SwinTransformerV2,
    "SwinMLP": SwinMLP,
}

__all__ = [
    "MODEL_REGISTRY", 
    "MLP", "UNet1d", "UNet2d", "UNet3d", 
    "FNO1d", "FNO2d", "FNO3d",
    "Transformer", "M2NO2d", "SwinTransformerV2", "SwinMLP"
    ]
