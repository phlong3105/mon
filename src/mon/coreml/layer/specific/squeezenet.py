#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for SqueezeNet
models.
"""

from __future__ import annotations

__all__ = [
    "Fire",
]

import torch
from torch import nn, Tensor

from mon.coreml import constant
from mon.coreml.layer import base, common


@constant.LAYER.register()
class Fire(base.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        squeeze_planes  : int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = common.Conv2d(
            in_channels  = in_channels,
            out_channels = squeeze_planes,
            kernel_size  = 1,
        )
        self.squeeze_activation = common.ReLU(inplace=True)
        self.expand1x1 = common.Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand1x1_planes,
            kernel_size  = 1,
        )
        self.expand1x1_activation = common.ReLU(inplace=True)
        self.expand3x3 = common.Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand3x3_planes,
            kernel_size  = 3,
            padding      = 1,
        )
        self.expand3x3_activation = common.ReLU(inplace=True)
        
    def forward(self, input: Tensor) -> Tensor:
        x     = input
        x     = self.squeeze_activation(self.squeeze(x))
        y_1x1 = self.expand1x1_activation(self.expand1x1(x))
        y_3x3 = self.expand3x3_activation(self.expand3x3(x))
        y     = torch.cat([y_1x1, y_3x3], dim=1)
        return y

    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        expand1x1_planes = args[2]
        expand3x3_planes = args[3]
        c2               = expand1x1_planes + expand3x3_planes
        ch.append(c2)
        return args, ch
