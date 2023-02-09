#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for DenseNet
models.
"""

from __future__ import annotations

__all__ = [
    "DenseBlock", "DenseLayer", "DenseTransition",
]

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional

from mon.coreml.layer import base, common
from mon.globals import LAYERS


@LAYERS.register()
class DenseLayer(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        bn_size         : int,
        drop_rate       : float,
        memory_efficient: bool  = False,
    ):
        super().__init__()
        self.norm1 = common.BatchNorm2d(in_channels)
        self.relu1 = common.ReLU(inplace=True)
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels * bn_size,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.norm2 = common.BatchNorm2d(out_channels * bn_size)
        self.relu2 = common.ReLU(inplace=True)
        self.conv2 = common.Conv2d(
            in_channels  = out_channels * bn_size,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            bias         = False
        )
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x = input
        x = [x] if isinstance(x, torch.Tensor) else x  # previous features
        x = torch.cat(x, dim=1)  # concat features
        y = self.conv1(self.relu1(self.norm1(x)))  # bottleneck
        y = self.conv2(self.relu2(self.norm2(y)))  # new features
        if self.drop_rate > 0.0:
            y = functional.dropout(
                input    = y,
                p        = self.drop_rate,
                training = self.training
            )
        return y


@LAYERS.register()
class DenseBlock(base.LayerParsingMixin, nn.ModuleDict):
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        num_layers      : int,
        bn_size         : int,
        drop_rate       : float,
        memory_efficient: bool  = False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels      = in_channels + i * out_channels,
                out_channels     = out_channels,
                bn_size          = bn_size,
                drop_rate        = drop_rate,
                memory_efficient = memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = [x]  # features
        for name, layer in self.items():
            new_features = layer(y)
            y.append(new_features)
        y = torch.cat(y, 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1           = args[0]
        out_channels = args[1]
        num_layers   = args[2]
        c2           = c1 + out_channels * num_layers
        ch.append(c2)
        return args, ch


@LAYERS.register()
class DenseTransition(base.LayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = common.BatchNorm2d(in_channels)
        self.relu = common.ReLU(inplace=True)
        self.conv = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            bias         = False,
        )
        self.pool = common.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.norm(x)
        y = self.relu(y)
        y = self.conv(y)
        y = self.pool(y)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[
        list, list]:
        c1 = args[0]
        c2 = c1 // 2
        ch.append(c2)
        return args, ch
