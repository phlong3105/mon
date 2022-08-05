#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolution + Normalization + Activation Layer.
"""

from __future__ import annotations

import math
from typing import Any
from typing import Optional

import torch
from torch import nn
from torch import Tensor

from one.core import Callable
from one.core import CONV_NORM_ACT_LAYERS
from one.core import Int2T
from one.core import Padding4T
from one.core import to_2tuple
from one.nn.layer.conv import create_conv2d
from one.nn.layer.conv import CrossConv
from one.nn.layer.norm_act import convert_norm_act
from one.nn.layer.padding import autopad

__all__ = [
    "ConvBnAct2d",
    "ConvBnMish2d",
    "ConvBnReLU2d",
    "ConvBnReLU62d",
    "CrossConvCSP",
    "DepthwiseConvBnMish2d",
    "GhostConv2d",
    "C3",
    "ConvBnAct",
    "ConvBnMish",
    "ConvBnReLU",
    "ConvBnReLU6",
    "DepthwiseConvBnMish",
    "GhostConv",
]


# MARK: - Modules

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_act2d")
class ConvBnAct2d(nn.Module):
    """Conv2d + BN + Act."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T             = (1, 1),
        padding     : Padding4T          = "",
        dilation    : Int2T             = (1, 1),
        groups      : int                = 1,
        bias        : bool               = False,
        padding_mode: str                = "zeros",
        device      : Any                = None,
        dtype       : Any                = None,
        apply_act   : bool               = True,
        act_layer   : Optional[Callable] = nn.ReLU,
        aa_layer    : Optional[Callable] = None,
        drop_block  : Optional[Callable] = None,
        **_
    ):
        super().__init__()
        use_aa      = aa_layer is not None
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        self.conv = create_conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = (1, 1) if use_aa else stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )

        # NOTE for backwards compatibility with models that use separate norm
        # and act layer definitions
        norm_act_layer = convert_norm_act(nn.BatchNorm2d, act_layer)
        self.bn = norm_act_layer(
            out_channels, apply_act=apply_act, drop_block=drop_block
        )
        self.aa = (aa_layer(channels=out_channels)
                   if stride == 2 and use_aa else None)

    # MARK: Properties

    @property
    def in_channels(self) -> int:
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.conv.out_channels

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.aa is not None:
            x = self.aa(x)
        return x


@CONV_NORM_ACT_LAYERS.register(name="conv_bn_mish2d")
class ConvBnMish2d(nn.Module):
    """Conv2d + BN + Mish."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T              = (1, 1),
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = None,
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        padding     = autopad(kernel_size, padding)
        
        self.conv = nn.Conv2d(
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
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.Mish() if apply_act else nn.Identity()
    
    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))
    

@CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu2d")
class ConvBnReLU2d(nn.Sequential):
    """Conv2d + BN + ReLU."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = 0,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("bn",  nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.ReLU() if apply_act else nn.Identity())


@CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu62d")
class ConvBnReLU62d(nn.Sequential):
    """Conv2d + BN + ReLU6."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        groups      : int                 = 1,
        bias        : bool                = False,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)

        self.add_module(
            "conv", nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = autopad(kernel_size, padding),
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.add_module("bn",  nn.BatchNorm2d(out_channels))
        self.add_module("act", nn.ReLU6() if apply_act else nn.Identity())


@CONV_NORM_ACT_LAYERS.register(name="cross_conv_csp")
class CrossConvCSP(nn.Module):
    """Cross Convolution CSP."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        groups      : int   = 1,
        expansion   : float = 0.5,
        shortcut    : bool  = True,
        **_
    ):
        super().__init__()
        c_       = int(out_channels * expansion)  # Hidden channels
        self.cv1 = ConvBnMish(in_channels, c_, 1, 1)
        self.cv2 = nn.Conv2d(in_channels,  c_, (1, 1), (1, 1), bias=False)
        self.cv3 = nn.Conv2d(c_, c_, (1, 1), (1, 1), bias=False)
        self.cv4 = ConvBnMish(2 * c_, out_channels, 1, 1)
        self.bn  = nn.BatchNorm2d(2 * c_)  # Applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m   = nn.Sequential(*[CrossConv(c_, c_, 3, 1, groups, 1.0, shortcut)
                                   for _ in range(number)])
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@CONV_NORM_ACT_LAYERS.register(name="depthwise_conv_bn_mish2d")
class DepthwiseConvBnMish2d(ConvBnMish2d):
    """Depthwise Conv2d + Bn + Mish.

    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (Int2T):
        
        stride (Int2T):
        
        padding (Padding4T, optional):
            Zero-padding added to both sides of the input. Default: `0`.
        dilation (Int2T):
            Spacing between kernel elements. Default: `(1, 1)`.
        bias (bool):
            Default: `True`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T              = (1, 1),
        stride      : Int2T              = (1, 1),
        padding     : Optional[Padding4T] = None,
        dilation    : Int2T              = (1, 1),
        bias        : bool                = True,
        padding_mode: str                 = None,
        device      : Any                 = None,
        dtype       : Any                 = None,
        apply_act   : bool                = True,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = math.gcd(in_channels, out_channels),
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
            apply_act    = apply_act,
        )
      

@CONV_NORM_ACT_LAYERS.register(name="ghost_conv2d")
class GhostConv2d(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T = 1,
        stride      : Int2T = 1,
        group       : int    = 1,
        apply_act   : bool   = True
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        c_ = out_channels // 2  # Hidden channels
        self.cv1 = ConvBnMish(in_channels, c_, kernel_size, stride, group,
                              apply_act=apply_act)
        self.cv2 = ConvBnMish(c_, c_, 5, 1, c_, apply_act=apply_act)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


# MARK: - Alias

C3                  = CrossConvCSP
ConvBnAct           = ConvBnAct2d
ConvBnMish          = ConvBnMish2d
ConvBnReLU          = ConvBnReLU2d
ConvBnReLU6         = ConvBnReLU62d
DepthwiseConvBnMish = DepthwiseConvBnMish2d
GhostConv           = GhostConv2d


# MARK: - Register

CONV_NORM_ACT_LAYERS.register(name="c3",                     module=C3)
CONV_NORM_ACT_LAYERS.register(name="conv_bn_act",            module=ConvBnAct)
CONV_NORM_ACT_LAYERS.register(name="conv_bn_mish",           module=ConvBnMish)
CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu",           module=ConvBnReLU)
CONV_NORM_ACT_LAYERS.register(name="conv_bn_relu6",          module=ConvBnReLU6)
CONV_NORM_ACT_LAYERS.register(name="depthwise_conv_bn_mish", module=DepthwiseConvBnMish)
CONV_NORM_ACT_LAYERS.register(name="ghost_conv",             module=GhostConv)
