#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the Zero-DCE model.
"""

from __future__ import annotations

__all__ = [
    "CalibrateNetwork",
    "EnhanceNetwork",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

class EnhanceNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, layers: int, channels: int):
        super().__init__()
        # Assign attributes
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        # Define layers
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + x
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


class CalibrateNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, layers: int, channels: int):
        super().__init__()
        # Assign attributes
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        # Define layers
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, 1, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = x - fea
        return delta

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
