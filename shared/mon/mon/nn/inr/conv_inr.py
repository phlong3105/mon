#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Conv-INR architecture.

References:
    - Paper: "Conv-INR: Convolutional Implicit Neural Representation for
      Multimodal Visual Signals," arXiv 2025.
"""

__all__ = [
    "ConvINR",
    "ConvINRLayer",
]

import torch
import torch.nn as nn


# ----- Layer -----
class ConvINRLayer(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(input)))


# ----- MLP -----
class ConvINR(nn.Module):
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        hidden_dim   : int = 32,
        hidden_layers: int = 10,
        kernel_size  : int = 3,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        # First layer
        self.net = []
        self.net.append(ConvINRLayer(in_channels, hidden_dim, kernel_size))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(ConvINRLayer(hidden_dim, hidden_dim, kernel_size))
        # Final layer
        self.net.append(nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=kernel_size // 2))
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        b, c, h, w = coords.shape
        if c > self.in_channels:
            coords = coords.view(b, c, h, w).permute(0, 3, 1, 2)  # B x H x W x C -> B x C x H x W
        output = self.net(coords)
        if c > self.in_channels:
            output = output.permute(0, 2, 3, 1)  # B x C x H x W -> B x H x W x C
        return output
