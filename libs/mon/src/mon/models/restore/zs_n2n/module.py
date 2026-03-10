#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the ZS-N2N model.
"""

from __future__ import annotations

__all__ = [
    "DenoiseNetwork",
    "ImprovedDenoiseNetwork",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region MODULES
# ==============================================================================

# --- ZS-N2N Network ---

class DenoiseNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Predicted noise tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        noise = self.conv3(y)
        return noise


# --- IZS-N2N Network ---

class GlobalContextBlock(nn.Module):
    """Global Context Self-Attention Mechanism.

    Enhance global semantic information before the convolutions.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
        """
        super().__init__()
        # Spatial pooling branch to compute attention matrix
        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Transform branch (Conv2d -> LayerNorm -> ReLU -> Conv2d)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1]), # Normalizes across channels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # 1. Calculate similarity/attention matrix
        context = self.context_conv(x).view(b, 1, h * w)
        context = F.softmax(context, dim=-1)

        # 2. Weight the average of the features
        x_reshaped = x.view(b, c, h * w)
        context_out = torch.bmm(
            x_reshaped, context.transpose(1, 2)
        ).view(b, c, 1, 1)

        # 3. Transform and add back to the original input via skip connection
        transform_out = self.transform(context_out)
        return x + transform_out


class ChannelAttentionModule(nn.Module):
    """Channel Attention Mechanism.

    Extract dependencies between channels using 1D convolution.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int = 3):
        """Initialize a new instance.

        Args:
            kernel_size (int, optional): Kernel size for the 1D convolution.
                Defaults to 3.
        """
        super().__init__()
        # Adaptive average pooling compresses each channel to one dimension
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x)                   # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)    # (B, 1, C) for Conv1D
        y = self.conv(y)                       # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        # Multiply weights with the corresponding elements of the feature map
        return x * self.sigmoid(y)


class ImprovedDenoiseNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
        """
        super().__init__()

        # 1. Global Context Module
        self.global_context = GlobalContextBlock(in_channels=in_channels)

        # 2. Noise Fitting Convolutions
        # Increases channels from 3 to 48 with 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Uses dilated convolution (dilation=2) to increase receptive field to 5x5
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Reduces channels back from 48 to 3 with 1x1 kernel
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

        # 3. Channel Attention Module
        self.channel_attention = ChannelAttentionModule()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Predicted noise tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # Step 1: Global Context
        y = self.global_context(x)

        # Step 2: Convolutional mapping
        y = self.lrelu1(self.conv1(y))
        y = self.lrelu2(self.conv2(y))
        y = self.conv3(y)

        # Step 3: Channel Attention
        noise = self.channel_attention(y)

        # Note: The network fits the noise parameter f_θ(y).
        # To get the denoised image during inference, you subtract this from the input: x = y - f_θ(y)
        return noise

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
