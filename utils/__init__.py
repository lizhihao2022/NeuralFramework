# utils/__init__.py

# ----------------------------------------------------------------------
# helper utilities
# ----------------------------------------------------------------------
from .helper import (
    set_seed,
    set_device,
    load_config,
    save_config,
    get_dir_path,
    set_up_logger,
    save_code,
)

# ----------------------------------------------------------------------
# distributed helpers
# ----------------------------------------------------------------------
from .ddp import (
    init_distributed,
    debug_barrier,
)

# ----------------------------------------------------------------------
# normalizers
# ----------------------------------------------------------------------
from .normalizer import (
    UnitGaussianNormalizer,
    GaussianNormalizer,
)

# ----------------------------------------------------------------------
# loss utilities
# ----------------------------------------------------------------------
from .loss import (
    LOSS_REGISTRY,
    register_loss,
    CompositeLoss,
    LpLoss,
    AverageRecord,
    LossRecord,
)

# ----------------------------------------------------------------------
# metrics utilities
# ----------------------------------------------------------------------
from .metrics import (
    mse,
    rmse,
    psnr,
    ssim,
    METRIC_REGISTRY,
    Evaluator,
)

# ----------------------------------------------------------------------
# Visualization utilities
# ----------------------------------------------------------------------
from .vis import (
    ns2d_vis,
    get_vis_fn,
)

# ----------------------------------------------------------------------
# Rollout utilities
# ----------------------------------------------------------------------
from .rollout import (
    autoregressive_rollout,
)

__all__ = [
    # helper
    "set_seed",
    "set_device",
    "load_config",
    "save_config",
    "get_dir_path",
    "set_up_logger",
    "save_code",
    # ddp
    "init_distributed",
    "debug_barrier",
    # normalizer
    "UnitGaussianNormalizer",
    "GaussianNormalizer",
    # loss
    "LOSS_REGISTRY",
    "register_loss",
    "CompositeLoss",
    "LpLoss",
    "AverageRecord",
    "LossRecord",
    # metrics
    "mse",
    "rmse",
    "psnr",
    "ssim",
    "METRIC_REGISTRY",
    "Evaluator",
    # vis
    "ns2d_vis",
    "get_vis_fn",
    # rollout
    "autoregressive_rollout",
]
