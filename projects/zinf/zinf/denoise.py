#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "Denoise",
]

import torch

from mon.core import nn


class Denoise(nn.Module):
    
    def __init__(self, in_channels: int, embed_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,    embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, in_channels,    1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
