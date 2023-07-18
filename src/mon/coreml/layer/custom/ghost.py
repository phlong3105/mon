#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Ghost modules."""

from __future__ import annotations

__all__ = [
    "GhostBottleneck", "GhostConv2d", "GhostSEBottleneck",
]

from typing import Any, Callable

import torch
from torch import nn

from mon.coreml.layer import base
from mon.coreml.layer.typing import _size_2_t
from mon.foundation import math
from mon.globals import LAYERS


# region Ghost Convolution

@LAYERS.register()
class GhostConv2d(base.ConvLayerParsingMixin, nn.Module):
    """GhostConv2d adopted from the paper: "GhostNet: More Features from Cheap
    Operations," CVPR 2020.
    
    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        ratio         : int                    = 2,
        kernel_size   : _size_2_t              = 1,
        dw_kernel_size: _size_2_t              = 3,
        stride        : _size_2_t              = 1,
        padding       : _size_2_t | str | None = None,
        dilation      : _size_2_t              = 1,
        groups        : int                    = 1,
        bias          : bool                   = True,
        padding_mode  : str                    = "zeros",
        device        : Any                    = None,
        dtype         : Any                    = None,
        act           : bool | Callable        = base.ReLU,
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels     = math.ceil(out_channels / ratio)
        new_channels      = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            base.Conv2d(
                in_channels  = in_channels,
                out_channels = init_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = kernel_size // 2,
                dilation     = dilation,
                groups       = groups,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            base.BatchNorm2d(init_channels),
            base.to_act_layer(act=act, inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            base.Conv2d(
                in_channels  = init_channels,
                out_channels = new_channels,
                kernel_size  = dw_kernel_size,
                stride       = 1,
                padding      = dw_kernel_size // 2,
                groups       = init_channels,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            base.BatchNorm2d(new_channels),
            base.to_act_layer(act=act, inplace=True),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y1 = self.primary_conv(x)
        y2 = self.cheap_operation(y1)
        y  = torch.cat([y1, y2], dim=1)
        y  = y[:, :self.out_channels, :, :]
        return y
    
# endregion


# region Ghost Bottleneck

@LAYERS.register()
class GhostSEBottleneck(base.PassThroughLayerParsingMixin, nn.Module):
    """Squeeze-and-Excite Bottleneck layer used in GhostBottleneck module."""
    
    def __init__(
        self,
        in_channels     : int,
        se_ratio        : float      = 0.25,
        reduced_base_chs: int | None = None,
        act             : Callable   = base.ReLU,
        gate_fn         : Callable   = base.hard_sigmoid,
        divisor         : int        = 4,
    ):
        super().__init__()
        self.gate_fn     = gate_fn
        reduced_channels = self.make_divisible(
            v       = (reduced_base_chs or in_channels) * se_ratio,
            divisor = divisor
        )
        self.avg_pool    = base.AdaptiveAvgPool2d(1)
        self.conv_reduce = base.Conv2d(
            in_channels  = in_channels,
            out_channels = reduced_channels,
            kernel_size  = 1,
            bias         = True
        )
        self.act         = base.to_act_layer(act=act, inplace=True)
        self.conv_expand = base.Conv2d(
            in_channels  = reduced_channels,
            out_channels = in_channels,
            kernel_size  = 1,
            bias         = True
        )
    
    def make_divisible(self, v, divisor, min_value=None):
        """This function is taken from the original tf repo. It ensures that
        all layers have a channel number that is divisible by 8 It can be seen
        here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.avg_pool(x)
        y = self.conv_reduce(y)
        y = self.act(y)
        y = self.conv_expand(y)
        y = x * self.gate_fn(y)
        return y


@LAYERS.register()
class GhostBottleneck(base.PassThroughLayerParsingMixin, nn.Module):
    """Ghost Bottleneck with optional SE.
    
    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """
    
    def __init__(
        self,
        in_channels : int,
        mid_channels: int,
        out_channels: int,
        kernel_size : _size_2_t       = 3,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        se_ratio    : float           = 0.0,
        act         : Callable        = base.ReLU,
    ):
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        
        # Point-wise expansion
        self.ghost1 = GhostConv2d(
            in_channels    = in_channels,
            out_channels   = mid_channels,
            kernel_size    = 1,
            dw_kernel_size = 3,
            stride         = stride,
            padding        = padding,
            dilation       = dilation,
            groups         = groups,
            bias           = bias,
            padding_mode   = padding_mode,
            device         = device,
            dtype          = dtype,
            act            = base.ReLU,
        )
        
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = base.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = (kernel_size - 1) // 2,
                dilation     = dilation,
                groups       = mid_channels,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
            self.bn_dw = base.BatchNorm2d(mid_channels)
        
        # Squeeze-and-excitation
        if has_se:
            self.se = GhostSEBottleneck(
                in_channels = mid_channels,
                se_ratio    = se_ratio
            )
        else:
            self.se = None
        
        # Point-wise linear projection
        self.ghost2 = GhostConv2d(
            in_channels    = mid_channels,
            out_channels   = out_channels,
            kernel_size    = 1,
            dw_kernel_size = 3,
            stride         = stride,
            padding        = padding,
            dilation       = dilation,
            groups         = groups,
            bias           = bias,
            padding_mode   = padding_mode,
            device         = device,
            dtype          = dtype,
            act            = base.ReLU,
        )
        
        # Shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = (kernel_size - 1) // 2,
                    dilation     = dilation,
                    groups       = in_channels,
                    bias         = False,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                ),
                base.BatchNorm2d(in_channels),
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    dilation     = dilation,
                    groups       = groups,
                    bias         = False,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                ),
                base.BatchNorm2d(out_channels),
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # 1st ghost bottleneck
        y = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            y = self.conv_dw(y)
            y = self.bn_dw(y)
        # Squeeze-and-excitation
        if self.se is not None:
            y = self.se(y)
        # 2nd ghost bottleneck
        y = self.ghost2(y)
        y = y + self.shortcut(x)
        return y

# endregion
