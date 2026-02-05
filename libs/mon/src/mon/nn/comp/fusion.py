#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Feature fusion layers.

This module provides various feature fusion layers used for fusing features
into a single feature map.
"""

from __future__ import annotations

__all__ = [
    "AFF",
    "DAF",
    "MS_CAM",
    "iAFF",
]

import torch
import torch.nn as nn


# ==============================================================================
# region LAYERS
# ==============================================================================

class DAF(nn.Module):
    """Direct-Add-Fuse (DAF) layer.

    Add the input tensor and the residual tensor element-wise.
    """

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
            residual: Residual tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        return x + residual


class MS_CAM(nn.Module):
    """Multi-Scale Channel Attention Module (MS-CAM) layer.

    Apply multiscale channel attention to the input tensor.

    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch

    Attributes:
        local_att: Local attention module.
        global_att: Global attention module.
        sigmoid: Sigmoid activation function.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initialize a new instance.

        Args:
            channels: Number of input channels. Defaults to 64.
            ratio: Reduction ratio for the intermediate channels. Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Output tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
        """
        x_l  = self.local_att(x)
        x_g  = self.global_att(x)
        x_lg = x_l + x_g
        w    = self.sigmoid(x_lg)
        return x * w


class AFF(nn.Module):
    """Attentional Feature Fusion (AFF) layer.

    Fuse the input tensor and the residual tensor using an attentional mechanism.

    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch

    Attributes:
        local_att: Local attention module.
        global_att: Global attention module.
        sigmoid: Sigmoid activation function.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initialize a new instance.

        Args:
            channels: Number of input channels. Defaults to 64.
            ratio: Reduction ratio for the intermediate channels. Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
            residual: Residual tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Fused output tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        """
        x_a  = x + residual
        x_l  = self.local_att(x_a)
        x_g  = self.global_att(x_a)
        x_lg = x_l + x_g
        w    = self.sigmoid(x_lg)
        # x_o = 2 * input * w + 2 * residual * (1 - w)
        # Simplified: 2 * (input * w + residual * (1 - w))
        return 2 * (x * w + residual * (1 - w))


class iAFF(nn.Module):
    """Iterative Attentional Feature Fusion (iAFF) layer.

    Fuse the input tensor and the residual tensor using an iterative
    attentional mechanism.

    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch

    Attributes:
        local_att: First local attention module.
        global_att: First global attention module.
        local_att2: Second local attention module.
        global_att2: Second global attention module.
        sigmoid: Sigmoid activation function.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initialize a new instance.

        Args:
            channels: Number of input channels. Defaults to 64.
            ratio: Reduction ratio for the intermediate channels. Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging from 0.0 to 1.0.
            residual: Residual tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Fused output tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        """
        x_a   = x + residual
        x_l1  = self.local_att(x_a)
        x_g1  = self.global_att(x_a)
        x_lg1 = x_l1 + x_g1
        w1    = self.sigmoid(x_lg1)
        x_i   = x * w1 + residual * (1 - w1)

        x_l2  = self.local_att2(x_i)
        x_g2  = self.global_att2(x_i)
        x_lg2 = x_l2 + x_g2
        w2    = self.sigmoid(x_lg2)
        return x * w2 + residual * (1 - w2)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
