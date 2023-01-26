#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for UNet models.
"""

from __future__ import annotations

__all__ = [
    "UNetBlock",
]

from typing import Any

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import Int2T


@constant.LAYER.register()
class UNetBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T       = 3,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 1,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = False,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.norm1 = common.BatchNorm2d(num_features=out_channels)
        self.relu1 = common.ReLU(inplace=True)
        self.conv2 = common.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.norm2 = common.BatchNorm2d(num_features=out_channels)
        self.relu2 = common.ReLU(inplace=True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu2(y)
        return y
