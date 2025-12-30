# models/unet/unet3d.py
from typing import Any
from collections import OrderedDict

import torch
import torch.nn as nn


class UNet3d(nn.Module):
    """
    3D U-Net backbone for PDE / volumetric forecasting.

    Assumptions:
      - Input:  x of shape (B, D, H, W, C_in)
      - Output: y of shape (B, D, H, W, C_out)
      - D, H, W must be divisible by 16 (4 levels of 2× downsampling).

    Architecture:
      enc1 -> enc2 -> enc3 -> enc4 -> bottleneck -> dec4 -> dec3 -> dec2 -> dec1.
    """

    def __init__(
        self,
        model_params: dict,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # ---- core channel configuration ----
        init_features: int = model_params.get("init_features", 32)
        in_channels: int = model_params.get("in_channels", 3)
        out_channels: int = model_params.get("out_channels", 1)

        # ---- normalization & activation configuration ----
        # norm: "batch" | "group" | "none"
        # activation: "relu" | "gelu" | "tanh"
        self.norm_type: str = model_params.get("norm", "batch")
        self.act_type: str = model_params.get("activation", "relu")

        features = init_features

        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # Decoder path
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block(features * 8 * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block(features * 4 * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block(features * 2 * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # Final 1×1×1 conv to target channels
        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # Initialize weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Helper factories for normalization and activation
    # ------------------------------------------------------------------
    def _make_norm(self, num_features: int) -> nn.Module:
        """Create a 3D normalization layer according to self.norm_type."""
        norm = self.norm_type.lower()
        if norm == "batch":
            return nn.BatchNorm3d(num_features)
        if norm == "group":
            # Choose a reasonable number of groups that divides num_features
            num_groups = min(8, num_features)
            while num_groups > 1 and (num_features % num_groups != 0):
                num_groups -= 1
            if num_groups <= 1:
                return nn.Identity()
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        if norm in ("none", "identity"):
            return nn.Identity()
        raise ValueError(f"Unsupported norm type: {self.norm_type}")

    def _make_act(self) -> nn.Module:
        """Create an activation layer according to self.act_type."""
        act = self.act_type.lower()
        if act == "relu":
            return nn.ReLU(inplace=True)
        if act == "gelu":
            return nn.GELU()
        if act == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation type: {self.act_type}")

    # ------------------------------------------------------------------
    # Building block: two Conv3d layers with norm + activation
    # ------------------------------------------------------------------
    def _block(
        self,
        in_channels: int,
        features: int,
        name: str,
    ) -> nn.Sequential:
        """
        3D U-Net block:
            Conv3d -> Norm -> Act -> Conv3d -> Norm -> Act
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", self._make_norm(features)),
                    (name + "_act1", self._make_act()),
                    (
                        name + "_conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", self._make_norm(features)),
                    (name + "_act2", self._make_act()),
                ]
            )
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, D, H, W, C_in).

        Returns:
            Tensor of shape (B, D, H, W, C_out).
        """
        assert x.dim() == 5, f"Expected (B, D, H, W, C), got shape {x.shape}"
        B, D, H, W, C_in = x.shape

        # 4 levels of 2x pooling → D/H/W / 16 must be integer
        assert D % 16 == 0 and H % 16 == 0 and W % 16 == 0, (
            f"D, H, W must be divisible by 16 for this UNet configuration, "
            f"got D={D}, H={H}, W={W}."
        )

        # (B, D, H, W, C_in) -> (B, C_in, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.conv(dec1)  # (B, C_out, D, H, W)

        # (B, C_out, D, H, W) -> (B, D, H, W, C_out)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        return out

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        """Kaiming initialization for Conv3d and reasonable defaults for Norm."""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
