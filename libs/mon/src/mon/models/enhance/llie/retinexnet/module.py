#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the RetinexNet model.
"""

from __future__ import annotations

__all__ = [
    "DecomNet",
    "EnhanceNet",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region MODULES
# ==============================================================================

class DecomNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        # Define layers
        self.net1_conv0 = nn.Conv2d(4, channels, kernel_size * 3, padding=4, padding_mode="replicate")
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode="replicate"),
            nn.ReLU()
        )
        self.net1_recon = nn.Conv2d(channels, 4, kernel_size, padding=1, padding_mode="replicate")

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_in = torch.cat((x_max, x), dim=1)
        x0 = self.net1_conv0(x_in)
        xs = self.net1_convs(x0)
        y = self.net1_recon(xs)
        R = torch.sigmoid(y[:, 0:3, :, :])
        L = torch.sigmoid(y[:, 3:4, :, :])
        return R, L


class EnhanceNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        # Define layers
        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channels, kernel_size, padding=1, padding_mode="replicate")
        self.net2_conv1_1 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_conv1_2 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_conv1_3 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode="replicate")
        self.net2_deconv1_1 = nn.Conv2d(channels * 2, channels, kernel_size, padding=1, padding_mode="replicate")
        self.net2_deconv1_2 = nn.Conv2d(channels * 2, channels, kernel_size, padding=1, padding_mode="replicate")
        self.net2_deconv1_3 = nn.Conv2d(channels * 2, channels, kernel_size, padding=1, padding_mode="replicate")
        self.net2_fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, padding=1, padding_mode="replicate")
        self.net2_output = nn.Conv2d(channels, 1, kernel_size=3, padding=0)

    def forward(self, R: Tensor, L: Tensor) -> Tensor:
        x_in = torch.cat((R, L), dim=1)
        x0 = self.net2_conv0_1(x_in)
        x1 = self.relu(self.net2_conv1_1(x0))
        x2 = self.relu(self.net2_conv1_2(x1))
        x3 = self.relu(self.net2_conv1_3(x2))

        x3_up = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]))
        x4 = self.relu(self.net2_deconv1_1(torch.cat((x3_up, x2), dim=1)))
        x4_up = F.interpolate(x4, size=(x1.size()[2], x1.size()[3]))
        x5 = self.relu(self.net2_deconv1_2(torch.cat((x4_up, x1), dim=1)))
        x5_up = F.interpolate(x5, size=(x0.size()[2], x0.size()[3]))
        x6 = self.relu(self.net2_deconv1_3(torch.cat((x5_up, x0), dim=1)))

        x4_rs = F.interpolate(x4, size=(R.size()[2], R.size()[3]))
        x5_rs = F.interpolate(x5, size=(R.size()[2], R.size()[3]))
        x7 = torch.cat((x4_rs, x5_rs, x6), dim=1)
        x8 = self.net2_fusion(x7)
        y = self.net2_output(x8)
        return y

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
