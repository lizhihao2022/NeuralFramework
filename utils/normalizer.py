# utils/normalizer.py
import torch
from torch import nn
from typing import Optional, Any


class UnitGaussianNormalizer(nn.Module):
    """
    Per-dimension Gaussian normalizer.

    x is typically shaped like:
        - (n_train, n)
        - (n_train, T, n)
        - (n_train, n, T)
    and we compute mean/std along dim=0.
    """

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        # Compute statistics along the training batch dimension
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x using stored mean/std.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx: Optional[Any] = None) -> torch.Tensor:
        """
        Inverse normalization.

        Args:
            x: tensor to be denormalized.
            sample_idx: optional indices to select a subset of mean/std.
                        This is kept for compatibility with existing code.
        """
        # Select statistics
        if sample_idx is None:
            std = self.std
            mean = self.mean
        else:
            # sample_idx might be a list/tuple of index tensors or a tensor itself.
            idx0 = sample_idx[0] if isinstance(sample_idx, (list, tuple)) else sample_idx

            if self.mean.dim() == idx0.dim():
                # Case: mean/std indexed directly by batch-like indices
                std = self.std[sample_idx]
                mean = self.mean[sample_idx]
            else:
                # Case: extra leading dimension, e.g. (T, n) vs (batch,)
                std = self.std[:, sample_idx]
                mean = self.mean[:, sample_idx]

        std = (std + self.eps).to(x.device)
        mean = mean.to(x.device)

        shape = x.shape
        B = shape[0]
        C = shape[-1]

        # Flatten all but batch and channel to apply broadcasting
        x_flat = x.view(B, -1, C)

        x_flat = x_flat * std + mean
        x = x_flat.view(*shape)
        return x


class GaussianNormalizer(nn.Module):
    """
    Global Gaussian normalizer using a single scalar mean/std.
    """

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        mean = torch.mean(x)
        std = torch.std(x)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize x using global mean/std.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor, sample_idx: Optional[Any] = None) -> torch.Tensor:
        """
        Inverse normalization.

        sample_idx is kept for API compatibility and ignored.
        """
        return x * (self.std + self.eps) + self.mean
