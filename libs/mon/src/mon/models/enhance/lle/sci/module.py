#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the SCI and SCI++
models.
"""

from __future__ import annotations

__all__ = [
    "CalibrateNetwork",
    "CalibrateNetworkPP",
    "EnhanceNetwork",
    "EnhanceNetwork_Ha",
    "EnhanceNetwork_Hb",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

# --- SCI ---

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

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)
        feat = self.out_conv(feat)

        illu = feat + x
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

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)

        feat = self.out_conv(feat)
        delta = x - feat
        return delta


# --- SCI++ ---

def default_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    bias: bool = False,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=(kernel_size // 2),
        bias=bias
    )


class EnhanceNetwork_Ha(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, layers: int, channels: int):
        super().__init__()
        # Assign attributes
        kernel_size = 3

        # Define layers
        self.in_conv = nn.Sequential(
            default_conv(3, channels, kernel_size, True),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            conv = nn.Sequential(
                default_conv(channels, channels, kernel_size, True),
                nn.ReLU()
            )
            self.blocks.append(conv)
        self.out_conv = nn.Sequential(
            default_conv(channels, 3, kernel_size, True),
            nn.Sigmoid()
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)
        feat = self.out_conv(feat)

        illu = feat + x
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


class EnhanceNetwork_Hb(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, layers: int, channels: int):
        super().__init__()
        # Assign attributes
        kernel_size = 3

        # Define layers
        self.in_conv = nn.Sequential(
            default_conv(3, channels, kernel_size, True),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            conv = nn.Sequential(
                default_conv(channels, channels, kernel_size, True),
                nn.ReLU()
            )
            self.blocks.append(conv)
        self.out_conv = nn.Sequential(
            default_conv(channels, 3, kernel_size, True),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)
        feat = self.out_conv(feat)
        return feat


class CalibrateNetworkPP(nn.Module):

    def __init__(self, layers: int, channels: int):
        super().__init__()
        # Assign attributes
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        # Define layers
        self.in_conv = nn.Sequential(
            default_conv(3, channels, kernel_size, True),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            convs = nn.Sequential(
                default_conv(channels, channels, kernel_size, True),
                nn.ReLU(),
                default_conv(channels, channels, kernel_size, True),
                nn.ReLU()
            )
            self.blocks.append(convs)

        self.out_conv = nn.Sequential(
            default_conv(channels, 3, kernel_size, True),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        feat = self.in_conv(x)
        for conv in self.blocks:
            feat = feat + conv(feat)

        feat = self.out_conv(feat)
        delta = x + feat
        return delta

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
