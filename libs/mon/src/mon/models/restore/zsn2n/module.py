#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the ZS-N2N model.
"""

from __future__ import annotations

__all__ = [
    "DenoiseNetwork",
]

from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

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
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.conv3(y)
        return y

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
