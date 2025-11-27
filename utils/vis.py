# utils/vis.py
import numpy as np
import torch

from typing import Optional, Union

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def ns2d_vis(raw_x: Union[torch.Tensor, np.ndarray],
             raw_y: Union[torch.Tensor, np.ndarray],
             pred_y: Union[torch.Tensor, np.ndarray], 
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             emin: Optional[float] = None, emax: Optional[float] = None, 
             dpi: Optional[int] = 100, save_path: Optional[str] = None,) -> None:
    """
    Visualize 2D Navier-Stokes prediction results.

    Args:
        raw_x: input tensor of shape (H, W)
        raw_y: ground truth tensor of shape (H, W)
        pred_y: predicted tensor of shape (H, W)
        save_path: if provided, save the figure to this path
        vmin: minimum value for color scale (if None, use min of raw_y and pred_y)
        vmax: maximum value for color scale (if None, use max of raw_y and pred_y)
        emax: maximum absolute error for error color scale (if None, use max abs error)
        dpi: resolution of the saved figure
    """
    if isinstance(raw_x, torch.Tensor):
        raw_x = raw_x.cpu().numpy()
    if isinstance(raw_y, torch.Tensor):
        raw_y = raw_y.cpu().numpy()
    if isinstance(pred_y, torch.Tensor):
        pred_y = pred_y.cpu().numpy()
    
    vmin = vmin if vmin is not None else raw_y.min().item()
    vmax = vmax if vmax is not None else raw_y.max().item()
    
    error_y = np.abs(pred_y - raw_y)
    emax = emax if emax is not None else error_y.max().item()
    emin = emin if emin is not None else error_y.min().item()
    
    fig, axs = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True, dpi=dpi)
    
    im1 = axs[0].imshow(raw_x[:, :], cmap='viridis')
    axs[0].set_title('Input (x)')
    axs[0].axis('off')
    
    axs[1].imshow(raw_y[:, :], cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('Ground Truth (y)')
    axs[1].axis('off')
    
    im2 = axs[2].imshow(pred_y[:, :], cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title('Prediction (y_pred)')
    axs[2].axis('off')
    
    im3 = axs[3].imshow(error_y[:, :], cmap='inferno', vmin=emin, vmax=emax)
    axs[3].set_title('Absolute Error |y - y_pred|')
    axs[3].axis('off')
    
    for ax, im in [(axs[0], im1), (axs[2], im2), (axs[3], im3)]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(im, cax=cax)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        
    plt.show()
