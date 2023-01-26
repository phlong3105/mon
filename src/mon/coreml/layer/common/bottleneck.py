#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements bottleneck layers."""

from __future__ import annotations

__all__ = [
    "Bottleneck", "GhostBottleneck", "GhostSEBottleneck",
]

from typing import Any

import torch
import torchvision
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base
from mon.coreml.layer.common import (
    activation, conv, normalization, pooling,
)
from mon.coreml.typing import CallableType, Int2T


# region Bottleneck

@constant.LAYER.register()
class Bottleneck(torchvision.models.resnet.Bottleneck, base.PassThroughLayerParsingMixin):
    pass
    
# endregion


# region Ghost Bottleneck

@constant.LAYER.register()
class GhostSEBottleneck(base.PassThroughLayerParsingMixin, nn.Module):
    """Squeeze-and-Excite Bottleneck layer used in GhostBottleneck module."""
    
    def __init__(
        self,
        in_channels     : int,
        se_ratio        : float               = 0.25,
        reduced_base_chs: int          | None = None,
        act             : CallableType | None = activation.ReLU,
        gate_fn         : CallableType | None = activation.hard_sigmoid,
        divisor         : int                 = 4,
        *args, **kwargs
    ):
        super().__init__()
        self.gate_fn     = gate_fn
        reduced_channels = self.make_divisible(
            v       = (reduced_base_chs or in_channels) * se_ratio,
            divisor = divisor
        )
        self.avg_pool    = pooling.AdaptiveAvgPool2d(1)
        self.conv_reduce = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = reduced_channels,
            kernel_size  = 1,
            bias         = True
        )
        self.act         = activation.to_act_layer(act=act, inplace=True)
        self.conv_expand = conv.Conv2d(
            in_channels  = reduced_channels,
            out_channels = in_channels,
            kernel_size  = 1,
            bias         = True
        )
    
    def make_divisible(self, v, divisor, min_value=None):
        """This function is taken from the original tf repo. It ensures that
        all layers have a channel number that is divisible by 8 It can be seen
        here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
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


@constant.LAYER.register()
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
        kernel_size : Int2T               = 3,
        stride      : Int2T               = 1,
        padding     : Int2T | str         = 0,
        dilation    : Int2T               = 1,
        groups      : int                 = 1,
        bias        : bool                = True,
        padding_mode: str                 = "zeros",
        device      : Any                 = None,
        dtype       : Any                 = None,
        se_ratio    : float               = 0.0,
        act         : CallableType | None = activation.ReLU,
        *args, **kwargs
    ):
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = conv.GhostConv2d(
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
            act            = activation.ReLU,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = conv.Conv2d(
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
            self.bn_dw = normalization.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if has_se:
            self.se = GhostSEBottleneck(in_channels=mid_channels, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = conv.GhostConv2d(
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
            act            = activation.ReLU,
        )
        
        # Shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                conv.Conv2d(
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
                normalization.BatchNorm2d(in_channels),
                conv.Conv2d(
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
                normalization.BatchNorm2d(out_channels),
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
