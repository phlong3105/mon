#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements attention layers."""

__all__ = [
    "SEBlock",
    "SimAM",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- SE -----
class SEBlock(nn.Module):
    """Squeeze and Excite module.
    
    Args:
        in_channels: Number of input channels.
        rd_ratio: Input channel reduction ratio.
    
    References:
        - Paper: "Squeeze-and-Excitation Networks," CVPR 2018.
        - Code: https://github.com/hujie-frank/SENet
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, int(in_channels * rd_ratio), 1, 1, bias=True)
        self.expand = nn.Conv2d(int(in_channels * rd_ratio), in_channels, 1, 1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x
    

# ----- Parameter-Free Attention -----
class SimAM(nn.Module):
    """Implement Simple, Parameter-Free Attention Module (SimAM).

    Args:
        e_lambda: Regularization parameter for energy. Default: ``1e-4``.

    References:
        - Code: https://github.com/ZjjConan/SimAM
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid  = nn.Sigmoid()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        n          = w * h - 1
        d          = (input - input.mean(dim=[2, 3], keepdim=True)).pow(2)  # [B, C, H, W]
        v          = d.sum(dim=[2, 3], keepdim=True) / n   # [B, C, 1, 1]
        e_inv      = d / (4 * (v + self.e_lambda)) + 0.5   # [B, C, H, W]
        return input * self.sigmoid(e_inv)
