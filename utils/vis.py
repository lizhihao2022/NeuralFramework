# utils/vis.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ns2d_vis(
    raw_x: Union[torch.Tensor, np.ndarray],
    raw_y: Union[torch.Tensor, np.ndarray],
    pred_y: Union[torch.Tensor, np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    emin: Optional[float] = None,
    emax: Optional[float] = None,
    dpi: int = 100,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize 2D Navier-Stokes prediction results.

    Args:
        raw_x: input field, (H, W)
        raw_y: ground truth field, (H, W)
        pred_y: predicted field, (H, W)
        save_path: if provided, save the figure to this path
        vmin/vmax: value color scale for raw_y and pred_y
        emin/emax: abs error color scale
        dpi: figure dpi
    """
    raw_x = _to_numpy(raw_x)
    raw_y = _to_numpy(raw_y)
    pred_y = _to_numpy(pred_y)

    vmin = vmin if vmin is not None else float(raw_y.min())
    vmax = vmax if vmax is not None else float(raw_y.max())

    error_y = np.abs(pred_y - raw_y)
    emin = emin if emin is not None else float(error_y.min())
    emax = emax if emax is not None else float(error_y.max())

    fig, axs = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True, dpi=dpi)

    im1 = axs[0].imshow(raw_x, cmap="viridis")
    axs[0].set_title("Input (x)")
    axs[0].axis("off")

    axs[1].imshow(raw_y, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("Ground Truth (y)")
    axs[1].axis("off")

    im2 = axs[2].imshow(pred_y, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[2].set_title("Prediction (y_pred)")
    axs[2].axis("off")

    im3 = axs[3].imshow(error_y, cmap="inferno", vmin=emin, vmax=emax)
    axs[3].set_title("Absolute Error |y - y_pred|")
    axs[3].axis("off")

    for ax, im in [(axs[0], im1), (axs[2], im2), (axs[3], im3)]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(im, cax=cax)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    plt.show()


# -----------------------------
# Dataset visualization registry
# -----------------------------
VisFn = Callable[..., None]

VIS_REGISTRY: Dict[str, VisFn] = {
    # 你项目里 data.name 可能是 ns2d / ns_2d / NS2D 等，统一兜底多注册几个 key
    "ns2d": ns2d_vis,
    "ns_2d": ns2d_vis,
    "navier_stokes_2d": ns2d_vis,
}


def get_vis_fn(dataset_name: str) -> Optional[VisFn]:
    """Return dataset-specific visualization function or None if not registered."""
    if dataset_name is None:
        return None
    key = str(dataset_name).lower()
    return VIS_REGISTRY.get(key, None)
