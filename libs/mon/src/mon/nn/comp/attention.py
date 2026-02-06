#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention layers.

This module provides various attention layers used for extracting features from
high-dimensional inputs.
"""

from __future__ import annotations

__all__ = [
    "SEBlock",
    "SimAM",
]

from torch import nn, Tensor


# ==============================================================================
# region LAYERS
# ==============================================================================

# --- Squeeze-and-Excitation ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block.

    Apply squeeze-and-excitation to the input tensor to adaptively recalibrate
    channel-wise feature responses.

    References:
        - Paper: "Squeeze-and-Excitation Networks," CVPR 2018.
        - Code: https://github.com/hujie-frank/SENet
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            rd_ratio (float): Reduction ratio for intermediate channels.
                Defaults to 0.0625.
        """
        super().__init__()
        # Assign attributes
        mid_channels = int(in_channels * rd_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, 1, bias=True)
        self.expand = nn.Conv2d(mid_channels, in_channels, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y = self.avg_pool(x)
        y = self.reduce(y)
        y = self.act(y)
        y = self.expand(y)
        y = self.sigmoid(y)
        return x * y


# --- Parameter-Free Attention ---

class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module (SimAM).

    Apply a simple, parameter-free attention mechanism to the input tensor.

    References:
        - Code: https://github.com/ZjjConan/SimAM
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, e_lambda: float = 1e-4):
        """Initialize a new instance.

        Args:
            e_lambda (float): Lambda parameter for the exponential term.
                Defaults to 1e-4.
        """
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        d = x_minus_mu.pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return x * self.sigmoid(e_inv)


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
