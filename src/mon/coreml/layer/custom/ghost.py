#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Ghost modules from GhostNet and GhostNetv2 models."""

from __future__ import annotations

__all__ = [
    "GhostConv2dV2", "GhostBottleneck", "GhostBottleneckSE", "GhostBottleneckV2",
    "GhostConv2d",
]

from typing import Callable

import torch
from mon.coreml.layer import base
from mon.coreml.layer.typing import _size_2_t
from mon.foundation import math
from mon.globals import LAYERS
from torch import nn
from torch.nn import functional as F


# region Ghost Convolution

@LAYERS.register()
class GhostConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Ghost Convolution 2d proposed in the from the paper: "`GhostNet: More
    Features from Cheap Operations <https://arxiv.org/pdf/1911.11907.pdf>`__"
    
    References:
        - https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        ratio         : int             = 2,
        kernel_size   : _size_2_t       = 1,
        dw_kernel_size: _size_2_t       = 3,
        stride        : _size_2_t       = 1,
        act           : bool | Callable = base.ReLU,
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
                bias         = False,
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


@LAYERS.register()
class GhostConv2dV2(base.ConvLayerParsingMixin, nn.Module):
    """Ghost Convolution 2d V2 proposed in the from the paper: "`GhostNetV2:
    Enhance Cheap Operation with Long-Range Attention
    <https://arxiv.org/pdf/2211.12905.pdf>`__"
    
    References:
        - https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch
    """
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        ratio         : int             = 2,
        kernel_size   : _size_2_t       = 1,
        dw_kernel_size: _size_2_t       = 3,
        stride        : _size_2_t       = 1,
        act           : bool | Callable = base.ReLU,
        mode          : str             = "original",
    ):
        super().__init__()
        self.mode    = mode
        self.gate_fn = base.Sigmoid()
        
        if self.mode in ["original"]:
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
                    bias         = False,
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
                ),
                base.BatchNorm2d(new_channels),
                base.to_act_layer(act=act, inplace=True),
            )
        elif self.mode in ["attn", "attention"]:
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
                    groups       = 1,
                    bias         = False,
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
                ),
                base.BatchNorm2d(new_channels),
                base.to_act_layer(act=act, inplace=True),
            )
            self.short_conv = nn.Sequential(
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = kernel_size // 2,
                    groups       = 1,
                    bias         = False,
                ),
                base.BatchNorm2d(out_channels),
                base.Conv2d(
                    in_channels  = out_channels,
                    out_channels = out_channels,
                    kernel_size  = (1, 5),
                    stride       = 1,
                    padding      = (0, 2),
                    groups       = out_channels,
                    bias         = False,
                ),
                base.BatchNorm2d(out_channels),
                base.Conv2d(
                    in_channels  = out_channels,
                    out_channels = out_channels,
                    kernel_size  = (5, 1),
                    stride       = 1,
                    padding      = (2, 0),
                    groups       = out_channels,
                    bias         = False,
                ),
                base.BatchNorm2d(out_channels),
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.mode in ["original"]:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            y  = torch.cat([x1, x2], dim=1)
            return y[:, :self.out_channels, :, :]
        elif self.mode in ["attn", "attention"]:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1  = self.primary_conv(x)
            x2  = self.cheap_operation(x1)
            y   = torch.cat([x1, x2], dim=1)
            return y[:, :self.out_channels, :, :] \
                * F.interpolate(self.gate_fn(res), size=(y.shape[-2], y.shape[-1]), mode="nearest")
    
# endregion


# region Ghost Bottleneck

@LAYERS.register()
class GhostBottleneckSE(base.PassThroughLayerParsingMixin, nn.Module):
    """Squeeze-and-Excite Bottleneck layer used in
    :class:`mon.coreml.layer.custom.ghost.GhostBottleneck` module.
    """
    
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
    """Ghost Bottleneck with optional SE proposed in the paper: "`GhostNet: More
    Features from Cheap Operations <https://arxiv.org/pdf/1911.11907.pdf>`__"
    
    References:
        - https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """
    
    def __init__(
        self,
        in_channels : int,
        mid_channels: int,
        out_channels: int,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        se_ratio    : float     = 0.0,
        act         : Callable  = base.ReLU,
    ):
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        
        # Point-wise expansion
        self.ghost1 = GhostConv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            act          = act,
        )
        
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = base.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = (kernel_size - 1) // 2,
                groups       = mid_channels,
                bias         = False,
            )
            self.bn_dw = base.BatchNorm2d(mid_channels)
        
        # Squeeze-and-excitation
        if has_se:
            self.se = GhostBottleneckSE(in_channels=mid_channels, se_ratio=se_ratio)
        else:
            self.se = None
        
        # Point-wise linear projection
        self.ghost2 = GhostConv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            act          = False,
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
                    groups       = in_channels,
                    bias         = False,
                ),
                base.BatchNorm2d(in_channels),
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    bias         = False,
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


@LAYERS.register()
class GhostBottleneckV2(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        mid_channels: int,
        out_channels: int,
        kernel_size : _size_2_t  = 3,
        stride      : _size_2_t  = 1,
        se_ratio    : float      = 0.0,
        act         : Callable   = base.ReLU,
        layer_id    : int | None = None,
    ):
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostConv2dV2(
                in_channels  = in_channels,
                out_channels = out_channels,
                act          = act,
                mode         = "original",
            )
        else:
            self.ghost1 = GhostConv2dV2(
                in_channels  = in_channels,
                out_channels = mid_channels,
                act          = act,
                mode         = "attn",
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = base.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = (kernel_size - 1) // 2,
                groups       = mid_channels,
                bias         = False,
            )
            self.bn_dw = base.BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if has_se:
            self.se = GhostBottleneckSE(in_channels=mid_channels, se_ratio=se_ratio)
        else:
            self.se = None
            
        self.ghost2 = GhostConv2dV2(
            in_channels  = mid_channels,
            out_channels = out_channels,
            act          = False,
            mode         = "original",
        )
        
        # shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = in_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = (kernel_size - 1) // 2,
                    groups       = in_channels,
                    bias         = False,
                ),
                base.BatchNorm2d(in_channels),
                base.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    bias         = False,
                ),
                base.BatchNorm2d(out_channels),
            )
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x        = input
        residual = x
        x        = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x  = self.ghost2(x)
        x += self.shortcut(residual)
        return x

# endregion
