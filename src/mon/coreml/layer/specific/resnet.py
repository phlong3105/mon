#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for ResNet models.
"""

from __future__ import annotations

__all__ = [
    "ResNetBasicBlock", "ResNetBlock", "ResNetBottleneck",
]

from typing import Type

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import CallableType


@constant.LAYER.register()
class ResNetBasicBlock(base.ConvLayerParsingMixin, nn.Module):
    
    expansion: int = 1

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int                 = 1,
        groups      : int                 = 1,
        dilation    : int                 = 1,
        base_width  : int                 = 64,
        downsample  : nn.Module    | None = None,
        norm        : CallableType | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm is None:
            norm = common.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "`BasicBlock` only supports `groups=1` and `base_width=64`"
            )
        if dilation > 1:
            raise NotImplementedError(
                "dilation > 1 not supported in `BasicBlock`"
            )
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1      = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn1        = norm(out_channels)
        self.relu       = common.ReLU(inplace=True)
        self.conv2      = common.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2        = norm(out_channels)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y  = self.conv1(x)
        y  = self.bn1(y)
        y  = self.relu(y)
        y  = self.conv2(y)
        y  = self.bn2(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        y  = self.relu(y)
        return y


@constant.LAYER.register()
class ResNetBottleneck(base.ConvLayerParsingMixin, nn.Module):
    """Bottleneck in torchvision places the stride for down-sampling at 3x3
    convolution(self.conv2) while original implementation places the stride at
    the first 1x1 convolution(self.conv1) according to "Deep residual learning
    for image recognition" https://arxiv.org/abs/1512.03385. This variant is
    also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    
    expansion: int = 4

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int                 = 1,
        groups      : int                 = 1,
        dilation    : int                 = 1,
        base_width  : int                 = 64,
        downsample  : nn.Module    | None = None,
        norm        : CallableType | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm is None:
            norm = common.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1      = common.Conv2d(
            in_channels  = in_channels,
            out_channels = width,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn1        = norm(width)
        self.conv2      = common.Conv2d(
            in_channels  = width,
            out_channels = width,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2        = norm(width)
        self.conv3      = common.Conv2d(
            in_channels  = width,
            out_channels = out_channels * self.expansion,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn3        = norm(out_channels * self.expansion)
        self.relu       = common.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y  = self.conv1(x)
        y  = self.bn1(y)
        y  = self.relu(y)
        y  = self.conv2(y)
        y  = self.bn2(y)
        y  = self.relu(y)
        y  = self.conv3(y)
        y  = self.bn3(y)
        if self.downsample is not None:
            x = self.downsample(x)
        y += x
        y  = self.relu(y)
        return y


@constant.LAYER.register()
class ResNetBlock(base.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        block        : Type[ResNetBasicBlock | ResNetBottleneck],
        num_blocks   : int,
        in_channels  : int,
        out_channels : int,
        stride       : int                 = 1,
        groups       : int                 = 1,
        dilation     : int                 = 1,
        base_width   : int                 = 64,
        dilate       : bool                = False,
        norm         : CallableType | None = common.BatchNorm2d,
        *args, **kwargs
    ):
        super().__init__()
        downsample    = None
        prev_dilation = dilation
        if dilate:
            dilation *= stride
            stride    = 1
        
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = torch.nn.Sequential(
                common.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels * block.expansion,
                    kernel_size  = 1,
                    stride       = stride,
                    bias         = False,
                ),
                norm(out_channels * block.expansion),
            )
      
        layers = []
        layers.append(
            block(
                in_channels  = in_channels,
                out_channels = out_channels,
                stride       = stride,
                groups       = groups,
                dilation     = prev_dilation,
                base_width   = base_width,
                downsample   = downsample,
                norm= norm,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels  = out_channels * block.expansion,
                    out_channels = out_channels,
                    stride       = 1,
                    groups       = groups,
                    dilation     = dilation,
                    base_width   = base_width,
                    downsample   = None,
                    norm= norm,
                )
            )
        self.convs = torch.nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.convs(x)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1 = args[2]
        c2 = args[3]
        ch.append(c2)
        return args, ch
