#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the HVI-CIDNet model.
"""

from __future__ import annotations

__all__ = [
    "DSConv",
]

from torch import nn, Tensor

from .utils import weights_init


# ==============================================================================
# region MODULES
# ==============================================================================

class DSConv(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.apply(weights_init)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H_in, W_in) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C_out, H_out, W_out) and values
                ranging from 0.0 to 1.0.
        """
        y = self.depth_conv(x)
        y = self.point_conv(y)
        return y

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
