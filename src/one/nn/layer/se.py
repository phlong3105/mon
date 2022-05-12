#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Squeeze-and-Excite layers.
"""

from __future__ import annotations

from torch import nn
from torch import Tensor

__all__ = [
    "SqueezeAndExciteLayer",
    "SELayer",
]


# MARK: - Modules

class SqueezeAndExciteLayer(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y          = self.avg_pool(x).view(b, c)
        y          = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# MARK: - Alias

SELayer = SqueezeAndExciteLayer
