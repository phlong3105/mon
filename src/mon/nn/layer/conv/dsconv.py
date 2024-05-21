#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "DSConv2d",
    "DSConv2dReLU",
    "DSConvAct2d",
    "DWConv2d",
    "DepthwiseConv2d",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU",
    "DepthwiseSeparableConvAct2d",
    "PWConv2d",
    "PointwiseConv2d",
]

from typing import Any

import torch
from torch import nn

from mon.core import _callable, _size_2_t
from mon.nn.layer import activation
from mon.nn.layer.conv import base


# region Depthwise Separable Convolution

class DepthwiseConv2d(nn.Module):
    """Depthwise Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.dw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        return y


class PointwiseConv2d(nn.Module):
    """Pointwise Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.pw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv(x)
        return y
    

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.dw_conv = base.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
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
            kernel_size  = 1,
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


class DepthwiseSeparableConvAct2d(nn.Module):
    """Depthwise Separable Conv2d + Activation."""
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : _size_2_t,
        stride        : _size_2_t       = 1,
        padding       : _size_2_t | str = 0,
        dilation      : _size_2_t       = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
        act_layer     : _callable       = nn.ReLU,
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = act_layer()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.ds_conv(x)
        y = self.act(y)
        return y


class DepthwiseSeparableConv2dReLU(nn.Module):
    """Depthwise Separable Conv2d ReLU."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = activation.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.ds_conv(x)
        y = self.act(y)
        return y


DWConv2d     = DepthwiseConv2d
PWConv2d     = PointwiseConv2d
DSConv2d     = DepthwiseSeparableConv2d
DSConvAct2d  = DepthwiseSeparableConvAct2d
DSConv2dReLU = DepthwiseSeparableConv2dReLU

# endregion
