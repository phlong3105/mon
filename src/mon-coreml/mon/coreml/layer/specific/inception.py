#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for Inception
models.
"""

from __future__ import annotations

__all__ = [
    "Inception", "InceptionA", "InceptionAux1", "InceptionAux2", "InceptionB",
    "InceptionBasicConv2d", "InceptionC", "InceptionD", "InceptionE",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from mon import core
from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import CallableType, Int2T


@constant.LAYER.register()
class InceptionBasicConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Conv2d + BN + ReLU."""

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 0,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = False,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
        apply_act   : bool        = True,
        eps         : float       = 0.001,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = core.to_2tuple(kernel_size)
        stride      = core.to_2tuple(stride)
        dilation    = core.to_2tuple(dilation)
        self.conv = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = common.to_same_padding(kernel_size, padding),
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.bn  = common.BatchNorm2d(out_channels, eps)
        self.act = common.ReLU()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.act(self.bn(self.conv(x)))
        return y


@constant.LAYER.register()
class Inception(base.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        ch1x1      : int,
        ch3x3red   : int,
        ch3x3      : int,
        ch5x5red   : int,
        ch5x5      : int,
        pool_proj  : int,
        conv       : CallableType | None = None,
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
            common.MaxPool2d(
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
    

@constant.LAYER.register()
class InceptionA(base.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 224  # + pool_features
    
    def __init__(
        self,
        in_channels  : int,
        pool_features: int,
        conv         : CallableType | None = None,
        *args, **kwargs
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
        x = input
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
    

@constant.LAYER.register()
class InceptionB(base.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 480   # + in_channels
    
    def __init__(
        self,
        in_channels: int,
        conv       : CallableType | None = None,
        *args, **kwargs
    ):
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
    

@constant.LAYER.register()
class InceptionC(base.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 768
    
    def __init__(
        self,
        in_channels : int,
        channels_7x7: int,
        conv        : CallableType | None = None,
        *args, **kwargs
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
            kernel_size  = (1 , 7),
            padding      = (0 , 3),
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
    

@constant.LAYER.register()
class InceptionD(base.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 512   # + in_channels
    
    def __init__(
        self,
        in_channels: int,
        conv       : CallableType | None = None,
        *args, **kwargs
    ):
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
        y_pool = functional.max_pool2d(x, kernel_size=3, stride=2)
        y      = torch.cat([y_3x3, y_7x7x3, y_pool], 1)
        return y

    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        c1 = args[0]
        c2 = cls.base_out_channels + c1
        ch.append(c2)
        return args, ch
    

@constant.LAYER.register()
class InceptionE(base.LayerParsingMixin, nn.Module):
    
    base_out_channels: int = 2048
    
    def __init__(
        self,
        in_channels: int,
        conv       : CallableType | None = None,
        *args, **kwargs
    ):
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
        x        = input
        y_1x1    = self.branch1x1(x)
        y_3x3    = self.branch3x3_1(x)
        y_3x3    = [
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
    

@constant.LAYER.register()
class InceptionAux1(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        conv        : CallableType | None = None,
        *args, **kwargs
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
        self.fc           = common.Linear(768, out_channels)
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


@constant.LAYER.register()
class InceptionAux2(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout     : float               = 0.7,
        conv        : CallableType | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
       
        self.conv = conv(
            in_channels  = in_channels,
            out_channels = 128,
            kernel_size  = 1,
        )
        self.fc1     = common.Linear(in_features=2048, out_features=1024)
        self.fc2     = common.Linear(in_features=1024, out_features=out_channels)
        self.dropout = common.Dropout(p=dropout)

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
