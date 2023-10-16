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

from mon.globals import LAYERS
from mon.nn.typing import _size_2_t
from mon.vision import core, nn

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region UNet Block

@LAYERS.register()
class UNetBlock(nn.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t       = 3,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 1,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = False,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
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
        self.norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
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
        self.norm2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu2(y)
        return y

# endregion
