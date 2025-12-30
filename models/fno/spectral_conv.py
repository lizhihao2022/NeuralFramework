# models/fno/spectral_conv.py
import math
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution layer used in FNO-1D.

    It applies a learnable linear mapping in Fourier space on the
    lowest `modes` Fourier modes and returns the inverse FFT.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        # Only store weights for the lowest `modes` frequencies (positive half).
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul1d(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication of Fourier coefficients.

        Args:
            input:  (B, C_in, L_modes)
            weight: (C_in, C_out, L_modes)

        Returns:
            out: (B, C_out, L_modes)
        """
        return torch.einsum("bcl, col -> bol", input, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)

        Returns:
            out: (B, C_out, L)
        """
        B, C, L = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, n=L, dim=-1, norm="ortho")  # (B, C_in, L_ft)
        L_ft = x_ft.size(-1)

        m = min(self.modes, L_ft)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            L_ft,
            device=x.device,
            dtype=torch.cfloat,
        )

        if m > 0:
            out_ft[:, :, :m] = self.compl_mul1d(
                x_ft[:, :, :m],
                self.weight[:, :, :m],
            )

        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=L, dim=-1, norm="ortho")
        return x
    

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer used in Fourier Neural Operator (FNO).

    This layer:
      1) Transforms the input to Fourier domain via rFFT.
      2) Keeps only the lowest (modes_x, modes_y) Fourier modes.
      3) Applies learned complex-valued weights in the spectral domain.
      4) Transforms back to physical space via inverse rFFT.

    Args:
        in_channels (int):  Number of input channels.
        out_channels (int): Number of output channels.
        modes_x (int):      Number of Fourier modes to keep in the vertical (H) direction.
        modes_y (int):      Number of Fourier modes to keep in the horizontal (W) direction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Scale helps keep parameter magnitudes controlled at init
        scale = 1.0 / (in_channels * out_channels)

        # Weights for positive-low and negative-low frequency bands in H dimension.
        # Shape: (in_channels, out_channels, modes_x, modes_y)
        self.weight_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, H, W).
        """
        B, C_in, H, W = x.shape

        # rFFT over spatial dimensions (H, W)
        # Shape: (B, C_in, H, W//2 + 1), complex tensor
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Allocate output in Fourier domain
        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        # How many modes can we actually use given the resolution?
        m1 = min(self.modes_x, H)
        m2 = min(self.modes_y, W // 2 + 1)

        # Positive-low frequencies in H
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bchw,cohw->bohw",
            x_ft[:, :, :m1, :m2],
            self.weight_pos[:, :, :m1, :m2],
        )

        # Negative-low frequencies in H
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bchw,cohw->bohw",
            x_ft[:, :, -m1:, :m2],
            self.weight_neg[:, :, :m1, :m2],
        )

        # Back to physical space
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x_out


class SpectralConv3d(nn.Module):
    """
    3D spectral convolution layer used in FNO-3D.

    We apply a learnable linear mapping in Fourier space on a set
    of low-frequency modes (modes1, modes2, modes3) and respect
    the rFFT symmetries: last dimension is half-spectrum, while
    we handle positive/negative corners in the first two dims.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        # We use four sets of weights to cover +/- along the first two
        # frequency dimensions; last dimension uses half-spectrum.
        self.weight1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weight3 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weight4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul3d(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication for 3D Fourier coefficients.

        Args:
            input:  (B, C_in, D, H, W)
            weight: (C_in, C_out, D, H, W)

        Returns:
            out:    (B, C_out, D, H, W)
        """
        return torch.einsum("bcdhw, codhw -> bodhw", input, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)

        Returns:
            out: (B, C_out, D, H, W)
        """
        B, C, D, H, W = x.shape

        # rFFT over the last 3 dimensions
        x_ft = torch.fft.rfftn(x, s=(D, H, W), dim=(-3, -2, -1), norm="ortho")
        D_ft, H_ft, W_ft = x_ft.shape[-3:]

        m1 = min(self.modes1, D_ft)
        m2 = min(self.modes2, H_ft)
        m3 = min(self.modes3, W_ft)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            D_ft,
            H_ft,
            W_ft,
            device=x.device,
            dtype=torch.cfloat,
        )

        if m1 > 0 and m2 > 0 and m3 > 0:
            # Top-left corner
            out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
                x_ft[:, :, :m1, :m2, :m3],
                self.weight1[:, :, :m1, :m2, :m3],
            )
            # Bottom-left (negative along D)
            out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, :m2, :m3],
                self.weight2[:, :, :m1, :m2, :m3],
            )
            # Top-right (negative along H)
            out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, :m1, -m2:, :m3],
                self.weight3[:, :, :m1, :m2, :m3],
            )
            # Bottom-right (negative along D and H)
            out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
                x_ft[:, :, -m1:, -m2:, :m3],
                self.weight4[:, :, :m1, :m2, :m3],
            )

        # Inverse rFFT
        x = torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1), norm="ortho")
        return x
