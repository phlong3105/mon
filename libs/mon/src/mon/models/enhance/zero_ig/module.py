#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the Zero-IG models.
"""

from __future__ import annotations

__all__ = [
    "Denoise1",
    "Denoise2",
    "Enhancer",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

class Denoise1(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, chan_embed: int = 48):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(3, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 3, 1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Denoise2(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, chan_embed: int = 96):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(6, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, 6, 1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class Enhancer(nn.Module):

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

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)
        feat = self.out_conv(feat)
        feat = torch.clamp(feat, 0.0001, 1)
        return feat

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
