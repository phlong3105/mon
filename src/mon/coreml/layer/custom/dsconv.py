#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Depthwise Separable Convolution modules."""

from __future__ import annotations

__all__ = [
    "DepthwiseSeparableConv2d", "DepthwiseSeparableConv2dReLU",
]

from typing import Any

import torch
from torch import nn

from mon.coreml.layer import base
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


# region Depthwise Separable Convolution

@LAYERS.register()
class DepthwiseSeparableConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d."""
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: _size_2_t,
        pw_kernel_size: _size_2_t,
        dw_stride     : _size_2_t       = 1,
        dw_padding    : _size_2_t | str = 0,
        pw_stride     : _size_2_t       = 1,
        pw_padding    : _size_2_t | str = 0,
        dilation      : _size_2_t       = 1,
        groups        : int             = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = dw_kernel_size,
            stride       = dw_stride,
            padding      = dw_padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            stride       = pw_stride,
            padding      = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.pw_conv(y)
        return y


@LAYERS.register()
class DepthwiseSeparableConv2dReLU(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d ReLU."""
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: _size_2_t,
        pw_kernel_size: _size_2_t,
        dw_stride     : _size_2_t       = 1,
        pw_stride     : _size_2_t       = 1,
        dw_padding    : _size_2_t | str = 0,
        pw_padding    : _size_2_t | str = 0,
        dilation      : _size_2_t       = 1,
        groups        : int             = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
    ):
        super().__init__()
        self.dw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = dw_kernel_size,
            stride       = dw_stride,
            padding      = dw_padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            stride       = pw_stride,
            padding      = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = base.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.pw_conv(y)
        y = self.act(y)
        return y

# endregion
