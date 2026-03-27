#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the PairLIE model.
"""

from __future__ import annotations

__all__ = [
    "L_Net",
    "R_Net",
    "N_Net",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

class L_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_channels: int = 64):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, 1, 3, 1, 0),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.L_net(x))


class R_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_channels: int = 64):
        super().__init__()
        # Define layers
        self.R_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, 3, 3, 1, 0),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.R_net(x))


class N_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_channels: int = 64):
        super().__init__()
        # Define layers
        self.N_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, num_channels, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_channels, 3, 3, 1, 0),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.N_net(x))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
