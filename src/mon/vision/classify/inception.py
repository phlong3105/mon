#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Inception models."""

from __future__ import annotations

__all__ = [
    "Inception",
]

from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional

from mon import nn
from mon.core import builtins, pathlib
from mon.globals import LAYERS, MODELS
from mon.nn import _size_2_t
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class InceptionBasicConv2d(nn.ConvLayerParsingMixin, nn.Module):
    """Conv2d + BN + ReLU."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = False,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        apply_act   : bool            = True,
        eps         : float           = 0.001,
    ):
        super().__init__()
        kernel_size = builtins.to_2tuple(kernel_size)
        stride      = builtins.to_2tuple(stride)
        dilation    = builtins.to_2tuple(dilation)
        self.conv   = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = nn.to_same_padding(kernel_size, padding),
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.bn  = nn.BatchNorm2d(out_channels, eps)
        self.act = nn.ReLU()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.act(self.bn(self.conv(x)))
        return y


@LAYERS.register()
class Inception(nn.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        ch1x1      : int,
        ch3x3red   : int,
        ch3x3      : int,
        ch5x5red   : int,
        ch5x5      : int,
        pool_proj  : int,
        conv       : Callable = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1 = conv(
            in_channels  = in_channels,
            out_channels = ch1x1,
            kernel_size  = 1,
        )
        self.branch2 = torch.nn.Sequential(
            conv(
                in_channels  = in_channels,
                out_channels = ch3x3red,
                kernel_size  = 1,
            ),
            conv(
                in_channels  = ch3x3red,
                out_channels = ch3x3,
                kernel_size  = 3,
                padding      = 1,
            )
        )
        self.branch3 = torch.nn.Sequential(
            conv(
                in_channels  = in_channels,
                out_channels = ch5x5red,
                kernel_size  = 1,
            ),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for
            # details.
            conv(
                in_channels  = ch5x5red,
                out_channels = ch5x5,
                kernel_size  = 3,
                padding      = 1,
            ),
        )
        self.branch4 = torch.nn.Sequential(
            nn.MaxPool2d(
                kernel_size = 3,
                stride      = 1,
                padding     = 1,
                ceil_mode   = True,
            ),
            conv(
                in_channels  = in_channels,
                out_channels = pool_proj,
                kernel_size  = 1,
            ),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        y  = torch.cat([y1, y2, y3, y4], dim=1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1        = args[0]
        ch1x1     = args[1]
        ch3x3     = args[3]
        ch5x5     = args[5]
        pool_proj = args[6]
        c2        = ch1x1 + ch3x3 + ch5x5 + pool_proj
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionA(nn.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 224  # + pool_features
    
    def __init__(
        self,
        in_channels  : int,
        pool_features: int,
        conv         : Callable = None,
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch5x5_1 = conv(
            in_channels  = in_channels,
            out_channels = 48,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch5x5_2 = conv(
            in_channels  = 48,
            out_channels = 64,
            kernel_size  = 5,
            padding      = 2,
            eps          = 0.001,
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 64,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
            eps          = 0.001,
        )
        self.branch3x3dbl_3 = conv(
            in_channels  = 96,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
            eps          = 0.001,
        )
        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = pool_features,
            kernel_size  = 1,
            eps          = 0.001,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x        = input
        y_1x1    = self.branch1x1(x)
        y_5x5    = self.branch5x5_1(x)
        y_5x5    = self.branch5x5_2(y_5x5)
        y_3x3dbl = self.branch3x3dbl_1(x)
        y_3x3dbl = self.branch3x3dbl_2(y_3x3dbl)
        y_3x3dbl = self.branch3x3dbl_3(y_3x3dbl)
        y_pool   = functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        y_pool   = self.branch_pool(y_pool)
        y        = torch.cat([y_1x1, y_5x5, y_3x3dbl, y_pool], 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = cls.base_out_channels + args[1]
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionB(nn.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 480  # + in_channels

    def __init__(self, in_channels: int, conv: Callable = None):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch3x3 = conv(
            in_channels  = in_channels,
            out_channels = 384,
            kernel_size  = 3,
            stride       = 2,
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 64,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
        )
        self.branch3x3dbl_3 = conv(
            in_channels  = 96,
            out_channels = 96,
            kernel_size  = 3,
            stride       = 2,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x           = input
        y_3x3       = self.branch3x3(x)
        y_3x3dbl    = self.branch3x3dbl_1(x)
        y_3x3dbl    = self.branch3x3dbl_2(y_3x3dbl)
        y_3x3dbl    = self.branch3x3dbl_3(y_3x3dbl)
        branch_pool = functional.max_pool2d(x, kernel_size=3, stride=2)
        y           = torch.cat([y_3x3, y_3x3dbl, branch_pool], 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1 = args[0]
        c2 = cls.base_out_channels + c1
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionC(nn.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 768
    
    def __init__(
        self,
        in_channels : int,
        channels_7x7: int,
        conv        : Callable = None,
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        c7 = channels_7x7
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch7x7_1 = conv(
            in_channels  = in_channels,
            out_channels = c7,
            kernel_size  = 1,
        )
        self.branch7x7_2 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7_3 = conv(
            in_channels  = c7,
            out_channels = 192,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = c7,
            kernel_size  = 1,
        )
        self.branch7x7dbl_2 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_3 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7dbl_4 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_5 = conv(
            in_channels  = c7,
            out_channels = 192,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x        = input
        y_1x1    = self.branch1x1(x)
        y_7x7    = self.branch7x7_1(x)
        y_7x7    = self.branch7x7_2(y_7x7)
        y_7x7    = self.branch7x7_3(y_7x7)
        y_7x7dbl = self.branch7x7dbl_1(x)
        y_7x7dbl = self.branch7x7dbl_2(y_7x7dbl)
        y_7x7dbl = self.branch7x7dbl_3(y_7x7dbl)
        y_7x7dbl = self.branch7x7dbl_4(y_7x7dbl)
        y_7x7dbl = self.branch7x7dbl_5(y_7x7dbl)
        y_pool   = functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        y_pool   = self.branch_pool(y_pool)
        y        = torch.cat([y_1x1, y_7x7, y_7x7dbl, y_pool], 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = cls.base_out_channels
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionD(nn.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 512  # + in_channels
    
    def __init__(self, in_channels: int, conv: Callable = None):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch3x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch3x3_2 = conv(
            in_channels  = 192,
            out_channels = 320,
            kernel_size  = 3,
            stride       = 2,
        )
        self.branch7x7x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch7x7x3_2 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7x3_3 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7x3_4 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = 3,
            stride       = 2,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x       = input
        y_3x3   = self.branch3x3_1(x)
        y_3x3   = self.branch3x3_2(y_3x3)
        y_7x7x3 = self.branch7x7x3_1(x)
        y_7x7x3 = self.branch7x7x3_2(y_7x7x3)
        y_7x7x3 = self.branch7x7x3_3(y_7x7x3)
        y_7x7x3 = self.branch7x7x3_4(y_7x7x3)
        y_pool  = functional.max_pool2d(x, kernel_size=3, stride=2)
        y       = torch.cat([y_3x3, y_7x7x3, y_pool], 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1 = args[0]
        c2 = cls.base_out_channels + c1
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionE(nn.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 2048
    
    def __init__(self, in_channels: int, conv: Callable = None):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 320,
            kernel_size  = 1,
        )
        self.branch3x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 384,
            kernel_size  = 1,
        )
        self.branch3x3_2a = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (1, 3),
            padding      = (0, 1),
        )
        self.branch3x3_2b = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (3, 1),
            padding      = (1, 0),
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 448,
            kernel_size  = 1,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 448,
            out_channels = 384,
            kernel_size  = 3,
            padding      = 1,
        )
        self.branch3x3dbl_3a = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (1, 3),
            padding      = (0, 1),
        )
        self.branch3x3dbl_3b = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (3, 1),
            padding      = (1, 0),
        )
        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        y_1x1 = self.branch1x1(x)
        y_3x3 = self.branch3x3_1(x)
        y_3x3 = [
            self.branch3x3_2a(y_3x3),
            self.branch3x3_2b(y_3x3),
        ]
        y_3x3    = torch.cat(y_3x3, 1)
        y_3x3dbl = self.branch3x3dbl_1(x)
        y_3x3dbl = self.branch3x3dbl_2(y_3x3dbl)
        y_3x3dbl = [
            self.branch3x3dbl_3a(y_3x3dbl),
            self.branch3x3dbl_3b(y_3x3dbl),
        ]
        y_3x3dbl = torch.cat(y_3x3dbl, 1)
        y_pool   = functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        y_pool   = self.branch_pool(y_pool)
        y        = torch.cat([y_1x1, y_3x3, y_3x3dbl, y_pool], 1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c2 = cls.base_out_channels
        ch.append(c2)
        return args, ch


@LAYERS.register()
class InceptionAux1(nn.HeadLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        conv        : Callable = None,
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.conv0 = conv(
            in_channels  = in_channels,
            out_channels = 128,
            kernel_size  = 1,
        )
        self.conv1 = conv(
            in_channels  = 128,
            out_channels = 768,
            kernel_size  = 5,
        )
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc           = nn.Linear(768, out_channels)
        self.fc.stddev    = 0.001  # type: ignore[assignment]
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # N x 768 x 17 x 17
        y = functional.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        y = self.conv0(y)
        # N x 128 x 5 x 5
        y = self.conv1(y)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        y = functional.adaptive_avg_pool2d(y, (1, 1))
        # N x 768 x 1 x 1
        y = torch.flatten(y, 1)
        # N x 768
        y = self.fc(y)
        # N x 1000
        return y


@LAYERS.register()
class InceptionAux2(nn.HeadLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout     : float    = 0.7,
        conv        : Callable = None,
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.conv = conv(
            in_channels  = in_channels,
            out_channels = 128,
            kernel_size  = 1,
        )
        self.fc1     = nn.Linear(in_features=2048, out_features=1024)
        self.fc2     = nn.Linear(in_features=1024, out_features=out_channels)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        y = functional.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        y = self.conv(y)
        # N x 128 x 4 x 4
        y = torch.flatten(y, 1)
        # N x 2048
        y = functional.relu(self.fc1(y), inplace=True)
        # N x 1024
        y = self.dropout(y)
        # N x 1024
        y = self.fc2(y)
        # N x 1000 (num_classes)
        return y
    
# endregion


# region Model

@MODELS.register(name="inception")
class Inception(base.ImageClassificationModel):
    """Inception.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
            "file_name"  : "inception3-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {
        "backbone": {
            "0.bn.bias"                         : "Conv2d_1a_3x3.bn.bias",
            "0.bn.running_mean"                 : "Conv2d_1a_3x3.bn.running_mean",
            "0.bn.running_var"                  : "Conv2d_1a_3x3.bn.running_var",
            "0.bn.weight"                       : "Conv2d_1a_3x3.bn.weight",
            "0.conv.weight"                     : "Conv2d_1a_3x3.conv.weight",
            "1.bn.bias"                         : "Conv2d_2a_3x3.bn.bias",
            "1.bn.running_mean"                 : "Conv2d_2a_3x3.bn.running_mean",
            "1.bn.running_var"                  : "Conv2d_2a_3x3.bn.running_var",
            "1.bn.weight"                       : "Conv2d_2a_3x3.bn.weight",
            "1.conv.weight"                     : "Conv2d_2a_3x3.conv.weight",
            "2.bn.bias"                         : "Conv2d_2b_3x3.bn.bias",
            "2.bn.running_mean"                 : "Conv2d_2b_3x3.bn.running_mean",
            "2.bn.running_var"                  : "Conv2d_2b_3x3.bn.running_var",
            "2.bn.weight"                       : "Conv2d_2b_3x3.bn.weight",
            "2.conv.weight"                     : "Conv2d_2b_3x3.conv.weight",
            "4.bn.bias"                         : "Conv2d_3b_1x1.bn.bias",
            "4.bn.running_mean"                 : "Conv2d_3b_1x1.bn.running_mean",
            "4.bn.running_var"                  : "Conv2d_3b_1x1.bn.running_var",
            "4.bn.weight"                       : "Conv2d_3b_1x1.bn.weight",
            "4.conv.weight"                     : "Conv2d_3b_1x1.conv.weight",
            "5.bn.bias"                         : "Conv2d_4a_3x3.bn.bias",
            "5.bn.running_mean"                 : "Conv2d_4a_3x3.bn.running_mean",
            "5.bn.running_var"                  : "Conv2d_4a_3x3.bn.running_var",
            "5.bn.weight"                       : "Conv2d_4a_3x3.bn.weight",
            "5.conv.weight"                     : "Conv2d_4a_3x3.conv.weight",
            "7.branch1x1.bn.bias"               : "Mixed_5b.branch1x1.bn.bias",
            "7.branch1x1.bn.running_mean"       : "Mixed_5b.branch1x1.bn.running_mean",
            "7.branch1x1.bn.running_var"        : "Mixed_5b.branch1x1.bn.running_var",
            "7.branch1x1.bn.weight"             : "Mixed_5b.branch1x1.bn.weight",
            "7.branch1x1.conv.weight"           : "Mixed_5b.branch1x1.conv.weight",
            "7.branch3x3dbl_1.bn.bias"          : "Mixed_5b.branch3x3dbl_1.bn.bias",
            "7.branch3x3dbl_1.bn.running_mean"  : "Mixed_5b.branch3x3dbl_1.bn.running_mean",
            "7.branch3x3dbl_1.bn.running_var"   : "Mixed_5b.branch3x3dbl_1.bn.running_var",
            "7.branch3x3dbl_1.bn.weight"        : "Mixed_5b.branch3x3dbl_1.bn.weight",
            "7.branch3x3dbl_1.conv.weight"      : "Mixed_5b.branch3x3dbl_1.conv.weight",
            "7.branch3x3dbl_2.bn.bias"          : "Mixed_5b.branch3x3dbl_2.bn.bias",
            "7.branch3x3dbl_2.bn.running_mean"  : "Mixed_5b.branch3x3dbl_2.bn.running_mean",
            "7.branch3x3dbl_2.bn.running_var"   : "Mixed_5b.branch3x3dbl_2.bn.running_var",
            "7.branch3x3dbl_2.bn.weight"        : "Mixed_5b.branch3x3dbl_2.bn.weight",
            "7.branch3x3dbl_2.conv.weight"      : "Mixed_5b.branch3x3dbl_2.conv.weight",
            "7.branch3x3dbl_3.bn.bias"          : "Mixed_5b.branch3x3dbl_3.bn.bias",
            "7.branch3x3dbl_3.bn.running_mean"  : "Mixed_5b.branch3x3dbl_3.bn.running_mean",
            "7.branch3x3dbl_3.bn.running_var"   : "Mixed_5b.branch3x3dbl_3.bn.running_var",
            "7.branch3x3dbl_3.bn.weight"        : "Mixed_5b.branch3x3dbl_3.bn.weight",
            "7.branch3x3dbl_3.conv.weight"      : "Mixed_5b.branch3x3dbl_3.conv.weight",
            "7.branch5x5_1.bn.bias"             : "Mixed_5b.branch5x5_1.bn.bias",
            "7.branch5x5_1.bn.running_mean"     : "Mixed_5b.branch5x5_1.bn.running_mean",
            "7.branch5x5_1.bn.running_var"      : "Mixed_5b.branch5x5_1.bn.running_var",
            "7.branch5x5_1.bn.weight"           : "Mixed_5b.branch5x5_1.bn.weight",
            "7.branch5x5_1.conv.weight"         : "Mixed_5b.branch5x5_1.conv.weight",
            "7.branch5x5_2.bn.bias"             : "Mixed_5b.branch5x5_2.bn.bias",
            "7.branch5x5_2.bn.running_mean"     : "Mixed_5b.branch5x5_2.bn.running_mean",
            "7.branch5x5_2.bn.running_var"      : "Mixed_5b.branch5x5_2.bn.running_var",
            "7.branch5x5_2.bn.weight"           : "Mixed_5b.branch5x5_2.bn.weight",
            "7.branch5x5_2.conv.weight"         : "Mixed_5b.branch5x5_2.conv.weight",
            "7.branch_pool.bn.bias"             : "Mixed_5b.branch_pool.bn.bias",
            "7.branch_pool.bn.running_mean"     : "Mixed_5b.branch_pool.bn.running_mean",
            "7.branch_pool.bn.running_var"      : "Mixed_5b.branch_pool.bn.running_var",
            "7.branch_pool.bn.weight"           : "Mixed_5b.branch_pool.bn.weight",
            "7.branch_pool.conv.weight"         : "Mixed_5b.branch_pool.conv.weight",
            "8.branch1x1.bn.bias"               : "Mixed_5c.branch1x1.bn.bias",
            "8.branch1x1.bn.running_mean"       : "Mixed_5c.branch1x1.bn.running_mean",
            "8.branch1x1.bn.running_var"        : "Mixed_5c.branch1x1.bn.running_var",
            "8.branch1x1.bn.weight"             : "Mixed_5c.branch1x1.bn.weight",
            "8.branch1x1.conv.weight"           : "Mixed_5c.branch1x1.conv.weight",
            "8.branch3x3dbl_1.bn.bias"          : "Mixed_5c.branch3x3dbl_1.bn.bias",
            "8.branch3x3dbl_1.bn.running_mean"  : "Mixed_5c.branch3x3dbl_1.bn.running_mean",
            "8.branch3x3dbl_1.bn.running_var"   : "Mixed_5c.branch3x3dbl_1.bn.running_var",
            "8.branch3x3dbl_1.bn.weight"        : "Mixed_5c.branch3x3dbl_1.bn.weight",
            "8.branch3x3dbl_1.conv.weight"      : "Mixed_5c.branch3x3dbl_1.conv.weight",
            "8.branch3x3dbl_2.bn.bias"          : "Mixed_5c.branch3x3dbl_2.bn.bias",
            "8.branch3x3dbl_2.bn.running_mean"  : "Mixed_5c.branch3x3dbl_2.bn.running_mean",
            "8.branch3x3dbl_2.bn.running_var"   : "Mixed_5c.branch3x3dbl_2.bn.running_var",
            "8.branch3x3dbl_2.bn.weight"        : "Mixed_5c.branch3x3dbl_2.bn.weight",
            "8.branch3x3dbl_2.conv.weight"      : "Mixed_5c.branch3x3dbl_2.conv.weight",
            "8.branch3x3dbl_3.bn.bias"          : "Mixed_5c.branch3x3dbl_3.bn.bias",
            "8.branch3x3dbl_3.bn.running_mean"  : "Mixed_5c.branch3x3dbl_3.bn.running_mean",
            "8.branch3x3dbl_3.bn.running_var"   : "Mixed_5c.branch3x3dbl_3.bn.running_var",
            "8.branch3x3dbl_3.bn.weight"        : "Mixed_5c.branch3x3dbl_3.bn.weight",
            "8.branch3x3dbl_3.conv.weight"      : "Mixed_5c.branch3x3dbl_3.conv.weight",
            "8.branch5x5_1.bn.bias"             : "Mixed_5c.branch5x5_1.bn.bias",
            "8.branch5x5_1.bn.running_mean"     : "Mixed_5c.branch5x5_1.bn.running_mean",
            "8.branch5x5_1.bn.running_var"      : "Mixed_5c.branch5x5_1.bn.running_var",
            "8.branch5x5_1.bn.weight"           : "Mixed_5c.branch5x5_1.bn.weight",
            "8.branch5x5_1.conv.weight"         : "Mixed_5c.branch5x5_1.conv.weight",
            "8.branch5x5_2.bn.bias"             : "Mixed_5c.branch5x5_2.bn.bias",
            "8.branch5x5_2.bn.running_mean"     : "Mixed_5c.branch5x5_2.bn.running_mean",
            "8.branch5x5_2.bn.running_var"      : "Mixed_5c.branch5x5_2.bn.running_var",
            "8.branch5x5_2.bn.weight"           : "Mixed_5c.branch5x5_2.bn.weight",
            "8.branch5x5_2.conv.weight"         : "Mixed_5c.branch5x5_2.conv.weight",
            "8.branch_pool.bn.bias"             : "Mixed_5c.branch_pool.bn.bias",
            "8.branch_pool.bn.running_mean"     : "Mixed_5c.branch_pool.bn.running_mean",
            "8.branch_pool.bn.running_var"      : "Mixed_5c.branch_pool.bn.running_var",
            "8.branch_pool.bn.weight"           : "Mixed_5c.branch_pool.bn.weight",
            "8.branch_pool.conv.weight"         : "Mixed_5c.branch_pool.conv.weight",
            "9.branch1x1.bn.bias"               : "Mixed_5d.branch1x1.bn.bias",
            "9.branch1x1.bn.running_mean"       : "Mixed_5d.branch1x1.bn.running_mean",
            "9.branch1x1.bn.running_var"        : "Mixed_5d.branch1x1.bn.running_var",
            "9.branch1x1.bn.weight"             : "Mixed_5d.branch1x1.bn.weight",
            "9.branch1x1.conv.weight"           : "Mixed_5d.branch1x1.conv.weight",
            "9.branch3x3dbl_1.bn.bias"          : "Mixed_5d.branch3x3dbl_1.bn.bias",
            "9.branch3x3dbl_1.bn.running_mean"  : "Mixed_5d.branch3x3dbl_1.bn.running_mean",
            "9.branch3x3dbl_1.bn.running_var"   : "Mixed_5d.branch3x3dbl_1.bn.running_var",
            "9.branch3x3dbl_1.bn.weight"        : "Mixed_5d.branch3x3dbl_1.bn.weight",
            "9.branch3x3dbl_1.conv.weight"      : "Mixed_5d.branch3x3dbl_1.conv.weight",
            "9.branch3x3dbl_2.bn.bias"          : "Mixed_5d.branch3x3dbl_2.bn.bias",
            "9.branch3x3dbl_2.bn.running_mean"  : "Mixed_5d.branch3x3dbl_2.bn.running_mean",
            "9.branch3x3dbl_2.bn.running_var"   : "Mixed_5d.branch3x3dbl_2.bn.running_var",
            "9.branch3x3dbl_2.bn.weight"        : "Mixed_5d.branch3x3dbl_2.bn.weight",
            "9.branch3x3dbl_2.conv.weight"      : "Mixed_5d.branch3x3dbl_2.conv.weight",
            "9.branch3x3dbl_3.bn.bias"          : "Mixed_5d.branch3x3dbl_3.bn.bias",
            "9.branch3x3dbl_3.bn.running_mean"  : "Mixed_5d.branch3x3dbl_3.bn.running_mean",
            "9.branch3x3dbl_3.bn.running_var"   : "Mixed_5d.branch3x3dbl_3.bn.running_var",
            "9.branch3x3dbl_3.bn.weight"        : "Mixed_5d.branch3x3dbl_3.bn.weight",
            "9.branch3x3dbl_3.conv.weight"      : "Mixed_5d.branch3x3dbl_3.conv.weight",
            "9.branch5x5_1.bn.bias"             : "Mixed_5d.branch5x5_1.bn.bias",
            "9.branch5x5_1.bn.running_mean"     : "Mixed_5d.branch5x5_1.bn.running_mean",
            "9.branch5x5_1.bn.running_var"      : "Mixed_5d.branch5x5_1.bn.running_var",
            "9.branch5x5_1.bn.weight"           : "Mixed_5d.branch5x5_1.bn.weight",
            "9.branch5x5_1.conv.weight"         : "Mixed_5d.branch5x5_1.conv.weight",
            "9.branch5x5_2.bn.bias"             : "Mixed_5d.branch5x5_2.bn.bias",
            "9.branch5x5_2.bn.running_mean"     : "Mixed_5d.branch5x5_2.bn.running_mean",
            "9.branch5x5_2.bn.running_var"      : "Mixed_5d.branch5x5_2.bn.running_var",
            "9.branch5x5_2.bn.weight"           : "Mixed_5d.branch5x5_2.bn.weight",
            "9.branch5x5_2.conv.weight"         : "Mixed_5d.branch5x5_2.conv.weight",
            "9.branch_pool.bn.bias"             : "Mixed_5d.branch_pool.bn.bias",
            "9.branch_pool.bn.running_mean"     : "Mixed_5d.branch_pool.bn.running_mean",
            "9.branch_pool.bn.running_var"      : "Mixed_5d.branch_pool.bn.running_var",
            "9.branch_pool.bn.weight"           : "Mixed_5d.branch_pool.bn.weight",
            "9.branch_pool.conv.weight"         : "Mixed_5d.branch_pool.conv.weight",
            "10.branch3x3.bn.bias"              : "Mixed_6a.branch3x3.bn.bias",
            "10.branch3x3.bn.running_mean"      : "Mixed_6a.branch3x3.bn.running_mean",
            "10.branch3x3.bn.running_var"       : "Mixed_6a.branch3x3.bn.running_var",
            "10.branch3x3.bn.weight"            : "Mixed_6a.branch3x3.bn.weight",
            "10.branch3x3.conv.weight"          : "Mixed_6a.branch3x3.conv.weight",
            "10.branch3x3dbl_1.bn.bias"         : "Mixed_6a.branch3x3dbl_1.bn.bias",
            "10.branch3x3dbl_1.bn.running_mean" : "Mixed_6a.branch3x3dbl_1.bn.running_mean",
            "10.branch3x3dbl_1.bn.running_var"  : "Mixed_6a.branch3x3dbl_1.bn.running_var",
            "10.branch3x3dbl_1.bn.weight"       : "Mixed_6a.branch3x3dbl_1.bn.weight",
            "10.branch3x3dbl_1.conv.weight"     : "Mixed_6a.branch3x3dbl_1.conv.weight",
            "10.branch3x3dbl_2.bn.bias"         : "Mixed_6a.branch3x3dbl_2.bn.bias",
            "10.branch3x3dbl_2.bn.running_mean" : "Mixed_6a.branch3x3dbl_2.bn.running_mean",
            "10.branch3x3dbl_2.bn.running_var"  : "Mixed_6a.branch3x3dbl_2.bn.running_var",
            "10.branch3x3dbl_2.bn.weight"       : "Mixed_6a.branch3x3dbl_2.bn.weight",
            "10.branch3x3dbl_2.conv.weight"     : "Mixed_6a.branch3x3dbl_2.conv.weight",
            "10.branch3x3dbl_3.bn.bias"         : "Mixed_6a.branch3x3dbl_3.bn.bias",
            "10.branch3x3dbl_3.bn.running_mean" : "Mixed_6a.branch3x3dbl_3.bn.running_mean",
            "10.branch3x3dbl_3.bn.running_var"  : "Mixed_6a.branch3x3dbl_3.bn.running_var",
            "10.branch3x3dbl_3.bn.weight"       : "Mixed_6a.branch3x3dbl_3.bn.weight",
            "10.branch3x3dbl_3.conv.weight"     : "Mixed_6a.branch3x3dbl_3.conv.weight",
            "11.branch1x1.bn.bias"              : "Mixed_6b.branch1x1.bn.bias",
            "11.branch1x1.bn.running_mean"      : "Mixed_6b.branch1x1.bn.running_mean",
            "11.branch1x1.bn.running_var"       : "Mixed_6b.branch1x1.bn.running_var",
            "11.branch1x1.bn.weight"            : "Mixed_6b.branch1x1.bn.weight",
            "11.branch1x1.conv.weight"          : "Mixed_6b.branch1x1.conv.weight",
            "11.branch7x7_1.bn.bias"            : "Mixed_6b.branch7x7_1.bn.bias",
            "11.branch7x7_1.bn.running_mean"    : "Mixed_6b.branch7x7_1.bn.running_mean",
            "11.branch7x7_1.bn.running_var"     : "Mixed_6b.branch7x7_1.bn.running_var",
            "11.branch7x7_1.bn.weight"          : "Mixed_6b.branch7x7_1.bn.weight",
            "11.branch7x7_1.conv.weight"        : "Mixed_6b.branch7x7_1.conv.weight",
            "11.branch7x7_2.bn.bias"            : "Mixed_6b.branch7x7_2.bn.bias",
            "11.branch7x7_2.bn.running_mean"    : "Mixed_6b.branch7x7_2.bn.running_mean",
            "11.branch7x7_2.bn.running_var"     : "Mixed_6b.branch7x7_2.bn.running_var",
            "11.branch7x7_2.bn.weight"          : "Mixed_6b.branch7x7_2.bn.weight",
            "11.branch7x7_2.conv.weight"        : "Mixed_6b.branch7x7_2.conv.weight",
            "11.branch7x7_3.bn.bias"            : "Mixed_6b.branch7x7_3.bn.bias",
            "11.branch7x7_3.bn.running_mean"    : "Mixed_6b.branch7x7_3.bn.running_mean",
            "11.branch7x7_3.bn.running_var"     : "Mixed_6b.branch7x7_3.bn.running_var",
            "11.branch7x7_3.bn.weight"          : "Mixed_6b.branch7x7_3.bn.weight",
            "11.branch7x7_3.conv.weight"        : "Mixed_6b.branch7x7_3.conv.weight",
            "11.branch7x7dbl_1.bn.bias"         : "Mixed_6b.branch7x7dbl_1.bn.bias",
            "11.branch7x7dbl_1.bn.running_mean" : "Mixed_6b.branch7x7dbl_1.bn.running_mean",
            "11.branch7x7dbl_1.bn.running_var"  : "Mixed_6b.branch7x7dbl_1.bn.running_var",
            "11.branch7x7dbl_1.bn.weight"       : "Mixed_6b.branch7x7dbl_1.bn.weight",
            "11.branch7x7dbl_1.conv.weight"     : "Mixed_6b.branch7x7dbl_1.conv.weight",
            "11.branch7x7dbl_2.bn.bias"         : "Mixed_6b.branch7x7dbl_2.bn.bias",
            "11.branch7x7dbl_2.bn.running_mean" : "Mixed_6b.branch7x7dbl_2.bn.running_mean",
            "11.branch7x7dbl_2.bn.running_var"  : "Mixed_6b.branch7x7dbl_2.bn.running_var",
            "11.branch7x7dbl_2.bn.weight"       : "Mixed_6b.branch7x7dbl_2.bn.weight",
            "11.branch7x7dbl_2.conv.weight"     : "Mixed_6b.branch7x7dbl_2.conv.weight",
            "11.branch7x7dbl_3.bn.bias"         : "Mixed_6b.branch7x7dbl_3.bn.bias",
            "11.branch7x7dbl_3.bn.running_mean" : "Mixed_6b.branch7x7dbl_3.bn.running_mean",
            "11.branch7x7dbl_3.bn.running_var"  : "Mixed_6b.branch7x7dbl_3.bn.running_var",
            "11.branch7x7dbl_3.bn.weight"       : "Mixed_6b.branch7x7dbl_3.bn.weight",
            "11.branch7x7dbl_3.conv.weight"     : "Mixed_6b.branch7x7dbl_3.conv.weight",
            "11.branch7x7dbl_4.bn.bias"         : "Mixed_6b.branch7x7dbl_4.bn.bias",
            "11.branch7x7dbl_4.bn.running_mean" : "Mixed_6b.branch7x7dbl_4.bn.running_mean",
            "11.branch7x7dbl_4.bn.running_var"  : "Mixed_6b.branch7x7dbl_4.bn.running_var",
            "11.branch7x7dbl_4.bn.weight"       : "Mixed_6b.branch7x7dbl_4.bn.weight",
            "11.branch7x7dbl_4.conv.weight"     : "Mixed_6b.branch7x7dbl_4.conv.weight",
            "11.branch7x7dbl_5.bn.bias"         : "Mixed_6b.branch7x7dbl_5.bn.bias",
            "11.branch7x7dbl_5.bn.running_mean" : "Mixed_6b.branch7x7dbl_5.bn.running_mean",
            "11.branch7x7dbl_5.bn.running_var"  : "Mixed_6b.branch7x7dbl_5.bn.running_var",
            "11.branch7x7dbl_5.bn.weight"       : "Mixed_6b.branch7x7dbl_5.bn.weight",
            "11.branch7x7dbl_5.conv.weight"     : "Mixed_6b.branch7x7dbl_5.conv.weight",
            "11.branch_pool.bn.bias"            : "Mixed_6b.branch_pool.bn.bias",
            "11.branch_pool.bn.running_mean"    : "Mixed_6b.branch_pool.bn.running_mean",
            "11.branch_pool.bn.running_var"     : "Mixed_6b.branch_pool.bn.running_var",
            "11.branch_pool.bn.weight"          : "Mixed_6b.branch_pool.bn.weight",
            "11.branch_pool.conv.weight"        : "Mixed_6b.branch_pool.conv.weight",
            "12.branch1x1.bn.bias"              : "Mixed_6c.branch1x1.bn.bias",
            "12.branch1x1.bn.running_mean"      : "Mixed_6c.branch1x1.bn.running_mean",
            "12.branch1x1.bn.running_var"       : "Mixed_6c.branch1x1.bn.running_var",
            "12.branch1x1.bn.weight"            : "Mixed_6c.branch1x1.bn.weight",
            "12.branch1x1.conv.weight"          : "Mixed_6c.branch1x1.conv.weight",
            "12.branch7x7_1.bn.bias"            : "Mixed_6c.branch7x7_1.bn.bias",
            "12.branch7x7_1.bn.running_mean"    : "Mixed_6c.branch7x7_1.bn.running_mean",
            "12.branch7x7_1.bn.running_var"     : "Mixed_6c.branch7x7_1.bn.running_var",
            "12.branch7x7_1.bn.weight"          : "Mixed_6c.branch7x7_1.bn.weight",
            "12.branch7x7_1.conv.weight"        : "Mixed_6c.branch7x7_1.conv.weight",
            "12.branch7x7_2.bn.bias"            : "Mixed_6c.branch7x7_2.bn.bias",
            "12.branch7x7_2.bn.running_mean"    : "Mixed_6c.branch7x7_2.bn.running_mean",
            "12.branch7x7_2.bn.running_var"     : "Mixed_6c.branch7x7_2.bn.running_var",
            "12.branch7x7_2.bn.weight"          : "Mixed_6c.branch7x7_2.bn.weight",
            "12.branch7x7_2.conv.weight"        : "Mixed_6c.branch7x7_2.conv.weight",
            "12.branch7x7_3.bn.bias"            : "Mixed_6c.branch7x7_3.bn.bias",
            "12.branch7x7_3.bn.running_mean"    : "Mixed_6c.branch7x7_3.bn.running_mean",
            "12.branch7x7_3.bn.running_var"     : "Mixed_6c.branch7x7_3.bn.running_var",
            "12.branch7x7_3.bn.weight"          : "Mixed_6c.branch7x7_3.bn.weight",
            "12.branch7x7_3.conv.weight"        : "Mixed_6c.branch7x7_3.conv.weight",
            "12.branch7x7dbl_1.bn.bias"         : "Mixed_6c.branch7x7dbl_1.bn.bias",
            "12.branch7x7dbl_1.bn.running_mean" : "Mixed_6c.branch7x7dbl_1.bn.running_mean",
            "12.branch7x7dbl_1.bn.running_var"  : "Mixed_6c.branch7x7dbl_1.bn.running_var",
            "12.branch7x7dbl_1.bn.weight"       : "Mixed_6c.branch7x7dbl_1.bn.weight",
            "12.branch7x7dbl_1.conv.weight"     : "Mixed_6c.branch7x7dbl_1.conv.weight",
            "12.branch7x7dbl_2.bn.bias"         : "Mixed_6c.branch7x7dbl_2.bn.bias",
            "12.branch7x7dbl_2.bn.running_mean" : "Mixed_6c.branch7x7dbl_2.bn.running_mean",
            "12.branch7x7dbl_2.bn.running_var"  : "Mixed_6c.branch7x7dbl_2.bn.running_var",
            "12.branch7x7dbl_2.bn.weight"       : "Mixed_6c.branch7x7dbl_2.bn.weight",
            "12.branch7x7dbl_2.conv.weight"     : "Mixed_6c.branch7x7dbl_2.conv.weight",
            "12.branch7x7dbl_3.bn.bias"         : "Mixed_6c.branch7x7dbl_3.bn.bias",
            "12.branch7x7dbl_3.bn.running_mean" : "Mixed_6c.branch7x7dbl_3.bn.running_mean",
            "12.branch7x7dbl_3.bn.running_var"  : "Mixed_6c.branch7x7dbl_3.bn.running_var",
            "12.branch7x7dbl_3.bn.weight"       : "Mixed_6c.branch7x7dbl_3.bn.weight",
            "12.branch7x7dbl_3.conv.weight"     : "Mixed_6c.branch7x7dbl_3.conv.weight",
            "12.branch7x7dbl_4.bn.bias"         : "Mixed_6c.branch7x7dbl_4.bn.bias",
            "12.branch7x7dbl_4.bn.running_mean" : "Mixed_6c.branch7x7dbl_4.bn.running_mean",
            "12.branch7x7dbl_4.bn.running_var"  : "Mixed_6c.branch7x7dbl_4.bn.running_var",
            "12.branch7x7dbl_4.bn.weight"       : "Mixed_6c.branch7x7dbl_4.bn.weight",
            "12.branch7x7dbl_4.conv.weight"     : "Mixed_6c.branch7x7dbl_4.conv.weight",
            "12.branch7x7dbl_5.bn.bias"         : "Mixed_6c.branch7x7dbl_5.bn.bias",
            "12.branch7x7dbl_5.bn.running_mean" : "Mixed_6c.branch7x7dbl_5.bn.running_mean",
            "12.branch7x7dbl_5.bn.running_var"  : "Mixed_6c.branch7x7dbl_5.bn.running_var",
            "12.branch7x7dbl_5.bn.weight"       : "Mixed_6c.branch7x7dbl_5.bn.weight",
            "12.branch7x7dbl_5.conv.weight"     : "Mixed_6c.branch7x7dbl_5.conv.weight",
            "12.branch_pool.bn.bias"            : "Mixed_6c.branch_pool.bn.bias",
            "12.branch_pool.bn.running_mean"    : "Mixed_6c.branch_pool.bn.running_mean",
            "12.branch_pool.bn.running_var"     : "Mixed_6c.branch_pool.bn.running_var",
            "12.branch_pool.bn.weight"          : "Mixed_6c.branch_pool.bn.weight",
            "12.branch_pool.conv.weight"        : "Mixed_6c.branch_pool.conv.weight",
            "13.branch1x1.bn.bias"              : "Mixed_6d.branch1x1.bn.bias",
            "13.branch1x1.bn.running_mean"      : "Mixed_6d.branch1x1.bn.running_mean",
            "13.branch1x1.bn.running_var"       : "Mixed_6d.branch1x1.bn.running_var",
            "13.branch1x1.bn.weight"            : "Mixed_6d.branch1x1.bn.weight",
            "13.branch1x1.conv.weight"          : "Mixed_6d.branch1x1.conv.weight",
            "13.branch7x7_1.bn.bias"            : "Mixed_6d.branch7x7_1.bn.bias",
            "13.branch7x7_1.bn.running_mean"    : "Mixed_6d.branch7x7_1.bn.running_mean",
            "13.branch7x7_1.bn.running_var"     : "Mixed_6d.branch7x7_1.bn.running_var",
            "13.branch7x7_1.bn.weight"          : "Mixed_6d.branch7x7_1.bn.weight",
            "13.branch7x7_1.conv.weight"        : "Mixed_6d.branch7x7_1.conv.weight",
            "13.branch7x7_2.bn.bias"            : "Mixed_6d.branch7x7_2.bn.bias",
            "13.branch7x7_2.bn.running_mean"    : "Mixed_6d.branch7x7_2.bn.running_mean",
            "13.branch7x7_2.bn.running_var"     : "Mixed_6d.branch7x7_2.bn.running_var",
            "13.branch7x7_2.bn.weight"          : "Mixed_6d.branch7x7_2.bn.weight",
            "13.branch7x7_2.conv.weight"        : "Mixed_6d.branch7x7_2.conv.weight",
            "13.branch7x7_3.bn.bias"            : "Mixed_6d.branch7x7_3.bn.bias",
            "13.branch7x7_3.bn.running_mean"    : "Mixed_6d.branch7x7_3.bn.running_mean",
            "13.branch7x7_3.bn.running_var"     : "Mixed_6d.branch7x7_3.bn.running_var",
            "13.branch7x7_3.bn.weight"          : "Mixed_6d.branch7x7_3.bn.weight",
            "13.branch7x7_3.conv.weight"        : "Mixed_6d.branch7x7_3.conv.weight",
            "13.branch7x7dbl_1.bn.bias"         : "Mixed_6d.branch7x7dbl_1.bn.bias",
            "13.branch7x7dbl_1.bn.running_mean" : "Mixed_6d.branch7x7dbl_1.bn.running_mean",
            "13.branch7x7dbl_1.bn.running_var"  : "Mixed_6d.branch7x7dbl_1.bn.running_var",
            "13.branch7x7dbl_1.bn.weight"       : "Mixed_6d.branch7x7dbl_1.bn.weight",
            "13.branch7x7dbl_1.conv.weight"     : "Mixed_6d.branch7x7dbl_1.conv.weight",
            "13.branch7x7dbl_2.bn.bias"         : "Mixed_6d.branch7x7dbl_2.bn.bias",
            "13.branch7x7dbl_2.bn.running_mean" : "Mixed_6d.branch7x7dbl_2.bn.running_mean",
            "13.branch7x7dbl_2.bn.running_var"  : "Mixed_6d.branch7x7dbl_2.bn.running_var",
            "13.branch7x7dbl_2.bn.weight"       : "Mixed_6d.branch7x7dbl_2.bn.weight",
            "13.branch7x7dbl_2.conv.weight"     : "Mixed_6d.branch7x7dbl_2.conv.weight",
            "13.branch7x7dbl_3.bn.bias"         : "Mixed_6d.branch7x7dbl_3.bn.bias",
            "13.branch7x7dbl_3.bn.running_mean" : "Mixed_6d.branch7x7dbl_3.bn.running_mean",
            "13.branch7x7dbl_3.bn.running_var"  : "Mixed_6d.branch7x7dbl_3.bn.running_var",
            "13.branch7x7dbl_3.bn.weight"       : "Mixed_6d.branch7x7dbl_3.bn.weight",
            "13.branch7x7dbl_3.conv.weight"     : "Mixed_6d.branch7x7dbl_3.conv.weight",
            "13.branch7x7dbl_4.bn.bias"         : "Mixed_6d.branch7x7dbl_4.bn.bias",
            "13.branch7x7dbl_4.bn.running_mean" : "Mixed_6d.branch7x7dbl_4.bn.running_mean",
            "13.branch7x7dbl_4.bn.running_var"  : "Mixed_6d.branch7x7dbl_4.bn.running_var",
            "13.branch7x7dbl_4.bn.weight"       : "Mixed_6d.branch7x7dbl_4.bn.weight",
            "13.branch7x7dbl_4.conv.weight"     : "Mixed_6d.branch7x7dbl_4.conv.weight",
            "13.branch7x7dbl_5.bn.bias"         : "Mixed_6d.branch7x7dbl_5.bn.bias",
            "13.branch7x7dbl_5.bn.running_mean" : "Mixed_6d.branch7x7dbl_5.bn.running_mean",
            "13.branch7x7dbl_5.bn.running_var"  : "Mixed_6d.branch7x7dbl_5.bn.running_var",
            "13.branch7x7dbl_5.bn.weight"       : "Mixed_6d.branch7x7dbl_5.bn.weight",
            "13.branch7x7dbl_5.conv.weight"     : "Mixed_6d.branch7x7dbl_5.conv.weight",
            "13.branch_pool.bn.bias"            : "Mixed_6d.branch_pool.bn.bias",
            "13.branch_pool.bn.running_mean"    : "Mixed_6d.branch_pool.bn.running_mean",
            "13.branch_pool.bn.running_var"     : "Mixed_6d.branch_pool.bn.running_var",
            "13.branch_pool.bn.weight"          : "Mixed_6d.branch_pool.bn.weight",
            "13.branch_pool.conv.weight"        : "Mixed_6d.branch_pool.conv.weight",
            "14.branch1x1.bn.bias"              : "Mixed_6e.branch1x1.bn.bias",
            "14.branch1x1.bn.running_mean"      : "Mixed_6e.branch1x1.bn.running_mean",
            "14.branch1x1.bn.running_var"       : "Mixed_6e.branch1x1.bn.running_var",
            "14.branch1x1.bn.weight"            : "Mixed_6e.branch1x1.bn.weight",
            "14.branch1x1.conv.weight"          : "Mixed_6e.branch1x1.conv.weight",
            "14.branch7x7_1.bn.bias"            : "Mixed_6e.branch7x7_1.bn.bias",
            "14.branch7x7_1.bn.running_mean"    : "Mixed_6e.branch7x7_1.bn.running_mean",
            "14.branch7x7_1.bn.running_var"     : "Mixed_6e.branch7x7_1.bn.running_var",
            "14.branch7x7_1.bn.weight"          : "Mixed_6e.branch7x7_1.bn.weight",
            "14.branch7x7_1.conv.weight"        : "Mixed_6e.branch7x7_1.conv.weight",
            "14.branch7x7_2.bn.bias"            : "Mixed_6e.branch7x7_2.bn.bias",
            "14.branch7x7_2.bn.running_mean"    : "Mixed_6e.branch7x7_2.bn.running_mean",
            "14.branch7x7_2.bn.running_var"     : "Mixed_6e.branch7x7_2.bn.running_var",
            "14.branch7x7_2.bn.weight"          : "Mixed_6e.branch7x7_2.bn.weight",
            "14.branch7x7_2.conv.weight"        : "Mixed_6e.branch7x7_2.conv.weight",
            "14.branch7x7_3.bn.bias"            : "Mixed_6e.branch7x7_3.bn.bias",
            "14.branch7x7_3.bn.running_mean"    : "Mixed_6e.branch7x7_3.bn.running_mean",
            "14.branch7x7_3.bn.running_var"     : "Mixed_6e.branch7x7_3.bn.running_var",
            "14.branch7x7_3.bn.weight"          : "Mixed_6e.branch7x7_3.bn.weight",
            "14.branch7x7_3.conv.weight"        : "Mixed_6e.branch7x7_3.conv.weight",
            "14.branch7x7dbl_1.bn.bias"         : "Mixed_6e.branch7x7dbl_1.bn.bias",
            "14.branch7x7dbl_1.bn.running_mean" : "Mixed_6e.branch7x7dbl_1.bn.running_mean",
            "14.branch7x7dbl_1.bn.running_var"  : "Mixed_6e.branch7x7dbl_1.bn.running_var",
            "14.branch7x7dbl_1.bn.weight"       : "Mixed_6e.branch7x7dbl_1.bn.weight",
            "14.branch7x7dbl_1.conv.weight"     : "Mixed_6e.branch7x7dbl_1.conv.weight",
            "14.branch7x7dbl_2.bn.bias"         : "Mixed_6e.branch7x7dbl_2.bn.bias",
            "14.branch7x7dbl_2.bn.running_mean" : "Mixed_6e.branch7x7dbl_2.bn.running_mean",
            "14.branch7x7dbl_2.bn.running_var"  : "Mixed_6e.branch7x7dbl_2.bn.running_var",
            "14.branch7x7dbl_2.bn.weight"       : "Mixed_6e.branch7x7dbl_2.bn.weight",
            "14.branch7x7dbl_2.conv.weight"     : "Mixed_6e.branch7x7dbl_2.conv.weight",
            "14.branch7x7dbl_3.bn.bias"         : "Mixed_6e.branch7x7dbl_3.bn.bias",
            "14.branch7x7dbl_3.bn.running_mean" : "Mixed_6e.branch7x7dbl_3.bn.running_mean",
            "14.branch7x7dbl_3.bn.running_var"  : "Mixed_6e.branch7x7dbl_3.bn.running_var",
            "14.branch7x7dbl_3.bn.weight"       : "Mixed_6e.branch7x7dbl_3.bn.weight",
            "14.branch7x7dbl_3.conv.weight"     : "Mixed_6e.branch7x7dbl_3.conv.weight",
            "14.branch7x7dbl_4.bn.bias"         : "Mixed_6e.branch7x7dbl_4.bn.bias",
            "14.branch7x7dbl_4.bn.running_mean" : "Mixed_6e.branch7x7dbl_4.bn.running_mean",
            "14.branch7x7dbl_4.bn.running_var"  : "Mixed_6e.branch7x7dbl_4.bn.running_var",
            "14.branch7x7dbl_4.bn.weight"       : "Mixed_6e.branch7x7dbl_4.bn.weight",
            "14.branch7x7dbl_4.conv.weight"     : "Mixed_6e.branch7x7dbl_4.conv.weight",
            "14.branch7x7dbl_5.bn.bias"         : "Mixed_6e.branch7x7dbl_5.bn.bias",
            "14.branch7x7dbl_5.bn.running_mean" : "Mixed_6e.branch7x7dbl_5.bn.running_mean",
            "14.branch7x7dbl_5.bn.running_var"  : "Mixed_6e.branch7x7dbl_5.bn.running_var",
            "14.branch7x7dbl_5.bn.weight"       : "Mixed_6e.branch7x7dbl_5.bn.weight",
            "14.branch7x7dbl_5.conv.weight"     : "Mixed_6e.branch7x7dbl_5.conv.weight",
            "14.branch_pool.bn.bias"            : "Mixed_6e.branch_pool.bn.bias",
            "14.branch_pool.bn.running_mean"    : "Mixed_6e.branch_pool.bn.running_mean",
            "14.branch_pool.bn.running_var"     : "Mixed_6e.branch_pool.bn.running_var",
            "14.branch_pool.bn.weight"          : "Mixed_6e.branch_pool.bn.weight",
            "14.branch_pool.conv.weight"        : "Mixed_6e.branch_pool.conv.weight",
            "15.conv0.bn.bias"                  : "AuxLogits.conv0.bn.bias",
            "15.conv0.bn.running_mean"          : "AuxLogits.conv0.bn.running_mean",
            "15.conv0.bn.running_var"           : "AuxLogits.conv0.bn.running_var",
            "15.conv0.bn.weight"                : "AuxLogits.conv0.bn.weight",
            "15.conv0.conv.weight"              : "AuxLogits.conv0.conv.weight",
            "15.conv1.bn.bias"                  : "AuxLogits.conv1.bn.bias",
            "15.conv1.bn.running_mean"          : "AuxLogits.conv1.bn.running_mean",
            "15.conv1.bn.running_var"           : "AuxLogits.conv1.bn.running_var",
            "15.conv1.bn.weight"                : "AuxLogits.conv1.bn.weight",
            "15.conv1.conv.weight"              : "AuxLogits.conv1.conv.weight",
            "15.fc.bias"                        : "AuxLogits.fc.bias",
            "15.fc.weight"                      : "AuxLogits.fc.weight",
            "16.branch3x3_1.bn.bias"            : "Mixed_7a.branch3x3_1.bn.bias",
            "16.branch3x3_1.bn.running_mean"    : "Mixed_7a.branch3x3_1.bn.running_mean",
            "16.branch3x3_1.bn.running_var"     : "Mixed_7a.branch3x3_1.bn.running_var",
            "16.branch3x3_1.bn.weight"          : "Mixed_7a.branch3x3_1.bn.weight",
            "16.branch3x3_1.conv.weight"        : "Mixed_7a.branch3x3_1.conv.weight",
            "16.branch3x3_2.bn.bias"            : "Mixed_7a.branch3x3_2.bn.bias",
            "16.branch3x3_2.bn.running_mean"    : "Mixed_7a.branch3x3_2.bn.running_mean",
            "16.branch3x3_2.bn.running_var"     : "Mixed_7a.branch3x3_2.bn.running_var",
            "16.branch3x3_2.bn.weight"          : "Mixed_7a.branch3x3_2.bn.weight",
            "16.branch3x3_2.conv.weight"        : "Mixed_7a.branch3x3_2.conv.weight",
            "16.branch7x7x3_1.bn.bias"          : "Mixed_7a.branch7x7x3_1.bn.bias",
            "16.branch7x7x3_1.bn.running_mean"  : "Mixed_7a.branch7x7x3_1.bn.running_mean",
            "16.branch7x7x3_1.bn.running_var"   : "Mixed_7a.branch7x7x3_1.bn.running_var",
            "16.branch7x7x3_1.bn.weight"        : "Mixed_7a.branch7x7x3_1.bn.weight",
            "16.branch7x7x3_1.conv.weight"      : "Mixed_7a.branch7x7x3_1.conv.weight",
            "16.branch7x7x3_2.bn.bias"          : "Mixed_7a.branch7x7x3_2.bn.bias",
            "16.branch7x7x3_2.bn.running_mean"  : "Mixed_7a.branch7x7x3_2.bn.running_mean",
            "16.branch7x7x3_2.bn.running_var"   : "Mixed_7a.branch7x7x3_2.bn.running_var",
            "16.branch7x7x3_2.bn.weight"        : "Mixed_7a.branch7x7x3_2.bn.weight",
            "16.branch7x7x3_2.conv.weight"      : "Mixed_7a.branch7x7x3_2.conv.weight",
            "16.branch7x7x3_3.bn.bias"          : "Mixed_7a.branch7x7x3_3.bn.bias",
            "16.branch7x7x3_3.bn.running_mean"  : "Mixed_7a.branch7x7x3_3.bn.running_mean",
            "16.branch7x7x3_3.bn.running_var"   : "Mixed_7a.branch7x7x3_3.bn.running_var",
            "16.branch7x7x3_3.bn.weight"        : "Mixed_7a.branch7x7x3_3.bn.weight",
            "16.branch7x7x3_3.conv.weight"      : "Mixed_7a.branch7x7x3_3.conv.weight",
            "16.branch7x7x3_4.bn.bias"          : "Mixed_7a.branch7x7x3_4.bn.bias",
            "16.branch7x7x3_4.bn.running_mean"  : "Mixed_7a.branch7x7x3_4.bn.running_mean",
            "16.branch7x7x3_4.bn.running_var"   : "Mixed_7a.branch7x7x3_4.bn.running_var",
            "16.branch7x7x3_4.bn.weight"        : "Mixed_7a.branch7x7x3_4.bn.weight",
            "16.branch7x7x3_4.conv.weight"      : "Mixed_7a.branch7x7x3_4.conv.weight",
            "17.branch1x1.bn.bias"              : "Mixed_7b.branch1x1.bn.bias",
            "17.branch1x1.bn.running_mean"      : "Mixed_7b.branch1x1.bn.running_mean",
            "17.branch1x1.bn.running_var"       : "Mixed_7b.branch1x1.bn.running_var",
            "17.branch1x1.bn.weight"            : "Mixed_7b.branch1x1.bn.weight",
            "17.branch1x1.conv.weight"          : "Mixed_7b.branch1x1.conv.weight",
            "17.branch3x3_1.bn.bias"            : "Mixed_7b.branch3x3_1.bn.bias",
            "17.branch3x3_1.bn.running_mean"    : "Mixed_7b.branch3x3_1.bn.running_mean",
            "17.branch3x3_1.bn.running_var"     : "Mixed_7b.branch3x3_1.bn.running_var",
            "17.branch3x3_1.bn.weight"          : "Mixed_7b.branch3x3_1.bn.weight",
            "17.branch3x3_1.conv.weight"        : "Mixed_7b.branch3x3_1.conv.weight",
            "17.branch3x3_2a.bn.bias"           : "Mixed_7b.branch3x3_2a.bn.bias",
            "17.branch3x3_2a.bn.running_mean"   : "Mixed_7b.branch3x3_2a.bn.running_mean",
            "17.branch3x3_2a.bn.running_var"    : "Mixed_7b.branch3x3_2a.bn.running_var",
            "17.branch3x3_2a.bn.weight"         : "Mixed_7b.branch3x3_2a.bn.weight",
            "17.branch3x3_2a.conv.weight"       : "Mixed_7b.branch3x3_2a.conv.weight",
            "17.branch3x3_2b.bn.bias"           : "Mixed_7b.branch3x3_2b.bn.bias",
            "17.branch3x3_2b.bn.running_mean"   : "Mixed_7b.branch3x3_2b.bn.running_mean",
            "17.branch3x3_2b.bn.running_var"    : "Mixed_7b.branch3x3_2b.bn.running_var",
            "17.branch3x3_2b.bn.weight"         : "Mixed_7b.branch3x3_2b.bn.weight",
            "17.branch3x3_2b.conv.weight"       : "Mixed_7b.branch3x3_2b.conv.weight",
            "17.branch3x3dbl_1.bn.bias"         : "Mixed_7b.branch3x3dbl_1.bn.bias",
            "17.branch3x3dbl_1.bn.running_mean" : "Mixed_7b.branch3x3dbl_1.bn.running_mean",
            "17.branch3x3dbl_1.bn.running_var"  : "Mixed_7b.branch3x3dbl_1.bn.running_var",
            "17.branch3x3dbl_1.bn.weight"       : "Mixed_7b.branch3x3dbl_1.bn.weight",
            "17.branch3x3dbl_1.conv.weight"     : "Mixed_7b.branch3x3dbl_1.conv.weight",
            "17.branch3x3dbl_2.bn.bias"         : "Mixed_7b.branch3x3dbl_2.bn.bias",
            "17.branch3x3dbl_2.bn.running_mean" : "Mixed_7b.branch3x3dbl_2.bn.running_mean",
            "17.branch3x3dbl_2.bn.running_var"  : "Mixed_7b.branch3x3dbl_2.bn.running_var",
            "17.branch3x3dbl_2.bn.weight"       : "Mixed_7b.branch3x3dbl_2.bn.weight",
            "17.branch3x3dbl_2.conv.weight"     : "Mixed_7b.branch3x3dbl_2.conv.weight",
            "17.branch3x3dbl_3a.bn.bias"        : "Mixed_7b.branch3x3dbl_3a.bn.bias",
            "17.branch3x3dbl_3a.bn.running_mean": "Mixed_7b.branch3x3dbl_3a.bn.running_mean",
            "17.branch3x3dbl_3a.bn.running_var" : "Mixed_7b.branch3x3dbl_3a.bn.running_var",
            "17.branch3x3dbl_3a.bn.weight"      : "Mixed_7b.branch3x3dbl_3a.bn.weight",
            "17.branch3x3dbl_3a.conv.weight"    : "Mixed_7b.branch3x3dbl_3a.conv.weight",
            "17.branch3x3dbl_3b.bn.bias"        : "Mixed_7b.branch3x3dbl_3b.bn.bias",
            "17.branch3x3dbl_3b.bn.running_mean": "Mixed_7b.branch3x3dbl_3b.bn.running_mean",
            "17.branch3x3dbl_3b.bn.running_var" : "Mixed_7b.branch3x3dbl_3b.bn.running_var",
            "17.branch3x3dbl_3b.bn.weight"      : "Mixed_7b.branch3x3dbl_3b.bn.weight",
            "17.branch3x3dbl_3b.conv.weight"    : "Mixed_7b.branch3x3dbl_3b.conv.weight",
            "17.branch_pool.bn.bias"            : "Mixed_7b.branch_pool.bn.bias",
            "17.branch_pool.bn.running_mean"    : "Mixed_7b.branch_pool.bn.running_mean",
            "17.branch_pool.bn.running_var"     : "Mixed_7b.branch_pool.bn.running_var",
            "17.branch_pool.bn.weight"          : "Mixed_7b.branch_pool.bn.weight",
            "17.branch_pool.conv.weight"        : "Mixed_7b.branch_pool.conv.weight",
            "18.branch1x1.bn.bias"              : "Mixed_7c.branch1x1.bn.bias",
            "18.branch1x1.bn.running_mean"      : "Mixed_7c.branch1x1.bn.running_mean",
            "18.branch1x1.bn.running_var"       : "Mixed_7c.branch1x1.bn.running_var",
            "18.branch1x1.bn.weight"            : "Mixed_7c.branch1x1.bn.weight",
            "18.branch1x1.conv.weight"          : "Mixed_7c.branch1x1.conv.weight",
            "18.branch3x3_1.bn.bias"            : "Mixed_7c.branch3x3_1.bn.bias",
            "18.branch3x3_1.bn.running_mean"    : "Mixed_7c.branch3x3_1.bn.running_mean",
            "18.branch3x3_1.bn.running_var"     : "Mixed_7c.branch3x3_1.bn.running_var",
            "18.branch3x3_1.bn.weight"          : "Mixed_7c.branch3x3_1.bn.weight",
            "18.branch3x3_1.conv.weight"        : "Mixed_7c.branch3x3_1.conv.weight",
            "18.branch3x3_2a.bn.bias"           : "Mixed_7c.branch3x3_2a.bn.bias",
            "18.branch3x3_2a.bn.running_mean"   : "Mixed_7c.branch3x3_2a.bn.running_mean",
            "18.branch3x3_2a.bn.running_var"    : "Mixed_7c.branch3x3_2a.bn.running_var",
            "18.branch3x3_2a.bn.weight"         : "Mixed_7c.branch3x3_2a.bn.weight",
            "18.branch3x3_2a.conv.weight"       : "Mixed_7c.branch3x3_2a.conv.weight",
            "18.branch3x3_2b.bn.bias"           : "Mixed_7c.branch3x3_2b.bn.bias",
            "18.branch3x3_2b.bn.running_mean"   : "Mixed_7c.branch3x3_2b.bn.running_mean",
            "18.branch3x3_2b.bn.running_var"    : "Mixed_7c.branch3x3_2b.bn.running_var",
            "18.branch3x3_2b.bn.weight"         : "Mixed_7c.branch3x3_2b.bn.weight",
            "18.branch3x3_2b.conv.weight"       : "Mixed_7c.branch3x3_2b.conv.weight",
            "18.branch3x3dbl_1.bn.bias"         : "Mixed_7c.branch3x3dbl_1.bn.bias",
            "18.branch3x3dbl_1.bn.running_mean" : "Mixed_7c.branch3x3dbl_1.bn.running_mean",
            "18.branch3x3dbl_1.bn.running_var"  : "Mixed_7c.branch3x3dbl_1.bn.running_var",
            "18.branch3x3dbl_1.bn.weight"       : "Mixed_7c.branch3x3dbl_1.bn.weight",
            "18.branch3x3dbl_1.conv.weight"     : "Mixed_7c.branch3x3dbl_1.conv.weight",
            "18.branch3x3dbl_2.bn.bias"         : "Mixed_7c.branch3x3dbl_2.bn.bias",
            "18.branch3x3dbl_2.bn.running_mean" : "Mixed_7c.branch3x3dbl_2.bn.running_mean",
            "18.branch3x3dbl_2.bn.running_var"  : "Mixed_7c.branch3x3dbl_2.bn.running_var",
            "18.branch3x3dbl_2.bn.weight"       : "Mixed_7c.branch3x3dbl_2.bn.weight",
            "18.branch3x3dbl_2.conv.weight"     : "Mixed_7c.branch3x3dbl_2.conv.weight",
            "18.branch3x3dbl_3a.bn.bias"        : "Mixed_7c.branch3x3dbl_3a.bn.bias",
            "18.branch3x3dbl_3a.bn.running_mean": "Mixed_7c.branch3x3dbl_3a.bn.running_mean",
            "18.branch3x3dbl_3a.bn.running_var" : "Mixed_7c.branch3x3dbl_3a.bn.running_var",
            "18.branch3x3dbl_3a.bn.weight"      : "Mixed_7c.branch3x3dbl_3a.bn.weight",
            "18.branch3x3dbl_3a.conv.weight"    : "Mixed_7c.branch3x3dbl_3a.conv.weight",
            "18.branch3x3dbl_3b.bn.bias"        : "Mixed_7c.branch3x3dbl_3b.bn.bias",
            "18.branch3x3dbl_3b.bn.running_mean": "Mixed_7c.branch3x3dbl_3b.bn.running_mean",
            "18.branch3x3dbl_3b.bn.running_var" : "Mixed_7c.branch3x3dbl_3b.bn.running_var",
            "18.branch3x3dbl_3b.bn.weight"      : "Mixed_7c.branch3x3dbl_3b.bn.weight",
            "18.branch3x3dbl_3b.conv.weight"    : "Mixed_7c.branch3x3dbl_3b.conv.weight",
            "18.branch_pool.bn.bias"            : "Mixed_7c.branch_pool.bn.bias",
            "18.branch_pool.bn.running_mean"    : "Mixed_7c.branch_pool.bn.running_mean",
            "18.branch_pool.bn.running_var"     : "Mixed_7c.branch_pool.bn.running_var",
            "18.branch_pool.bn.weight"          : "Mixed_7c.branch_pool.bn.weight",
            "18.branch_pool.conv.weight"        : "Mixed_7c.branch_pool.conv.weight",
        },
        "head"    : {
            "19.fc.bias"  : "fc.bias",
            "19.fc.weight": "fc.weight",
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "inception3.yaml",
            "name"   : "inception",
            "variant": "inception3"
        }
        super().__init__(*args, **kwargs)
    
    def init_weights(self, m: torch.nn.Module):
        classname    = m.__class__.__name__
        init_weights = self.config["zero_init_residual"]
        if init_weights:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias,   0)
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["imagenet"]:
            state_dict = nn.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
        
# endregion
