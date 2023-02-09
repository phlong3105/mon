#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for Zero-ADCE
models.
"""

from __future__ import annotations

__all__ = [
    "ABSConv2dS1", "ABSConv2dS2", "ABSConv2dS3", "ABSConv2dS4", "ABSConv2dS5",
    "ABSConv2dS6", "ABSConv2dS7", "ABSConv2dS8", "ABSConv2dS9", "ABSConv2dS10",
    "ABSConv2dS11", "ABSConv2dS12", "ABSConv2dS13", "ADCE",
    "AttentionSubspaceBlueprintSeparableConv2d1",
    "AttentionSubspaceBlueprintSeparableConv2d2",
    "AttentionSubspaceBlueprintSeparableConv2d3",
    "AttentionSubspaceBlueprintSeparableConv2d4",
    "AttentionSubspaceBlueprintSeparableConv2d5",
    "AttentionSubspaceBlueprintSeparableConv2d6",
    "AttentionSubspaceBlueprintSeparableConv2d7",
    "AttentionSubspaceBlueprintSeparableConv2d8",
    "AttentionSubspaceBlueprintSeparableConv2d9",
    "AttentionSubspaceBlueprintSeparableConv2d10",
    "AttentionSubspaceBlueprintSeparableConv2d11",
    "AttentionSubspaceBlueprintSeparableConv2d12",
    "AttentionSubspaceBlueprintSeparableConv2d13",
]

from typing import Any, Callable

import torch

from mon.coreml.layer import common
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d1(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.pw_conv1(x)
        # x = self.simam(x)
        # if self.act1 is not None:
        #     x = self.act1(x)
        x = self.pw_conv2(x)
        # if self.act2 is not None:
        #     x = self.act2(x)
        x = self.dw_conv(x)
        return x


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d2(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # y = self.simam(y)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        # if self.act2 is not None:
        #    y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d3(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # y = self.simam(y)
        # if self.act1 is not None:
        #    y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d4(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # y = self.simam(y)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d5(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        y = self.simam(y)
        # if self.act1 is not None:
        #    y = self.act1(y)
        y = self.pw_conv2(y)
        # if self.act2 is not None:
        #    y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d6(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.act1 is not None:
        #    y = self.act1(y)
        y = self.pw_conv2(y)
        y = self.simam(y)
        # if self.act2 is not None:
        #    y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d7(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.act1 is not None:
        #    y = self.act1(y)
        y = self.pw_conv2(y)
        # if self.act2 is not None:
        #    y = self.act2(y)
        y = self.dw_conv(y)
        y = self.simam(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d8(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        y = self.simam(y)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        # if self.act2 is not None:
        #    y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d9(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        y = self.simam(y)
        # if self.act1 is not None:
        #     y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d10(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        y = self.simam(y)
        # if self.act2 is not None:
        #     y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d11(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        # if self.act1 is not None:
        #     y = self.act1(y)
        y = self.pw_conv2(y)
        y = self.simam(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d12(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        y = self.simam(y)
        return y


@LAYERS.register()
class AttentionSubspaceBlueprintSeparableConv2d13(
    common.AttentionSubspaceBlueprintSeparableConv2d
):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        y = self.simam(y)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y


@LAYERS.register()
class ADCE(torch.nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 3,
        out_channels: int       = 3,
        mid_channels: int       = 32,
        conv        : Callable  = common.BSConv2dS,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 1,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
    ):
        super().__init__()
        self.downsample = common.Downsample(None, 1, "bilinear")
        self.upsample   = common.UpsamplingBilinear2d(None, 1)
        self.relu       = common.ReLU(inplace=True)
        self.conv1 = conv(
            in_channels  = in_channels,
            out_channels = mid_channels,
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
        self.conv2 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
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
        self.conv3 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
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
        self.conv4 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
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
        self.conv5 = conv(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
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
        self.conv6 = conv(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
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
        self.conv7 = common.Conv2d(
            in_channels  = mid_channels * 2,
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
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        x  = self.downsample(x)
        y1 = self.relu(self.conv1(x))
        y2 = self.relu(self.conv2(y1))
        y3 = self.relu(self.conv3(y2))
        y4 = self.relu(self.conv4(y3))
        y5 = self.relu(self.conv5(torch.cat([y3, y4], dim=1)))
        y6 = self.relu(self.conv6(torch.cat([y2, y5], dim=1)))
        y  = torch.tanh(self.conv7(torch.cat([y1, y6], dim=1)))
        y  = self.upsample(y)
        return y


ABSConv2dS1  = AttentionSubspaceBlueprintSeparableConv2d1
ABSConv2dS2  = AttentionSubspaceBlueprintSeparableConv2d2
ABSConv2dS3  = AttentionSubspaceBlueprintSeparableConv2d3
ABSConv2dS4  = AttentionSubspaceBlueprintSeparableConv2d4
ABSConv2dS5  = AttentionSubspaceBlueprintSeparableConv2d5
ABSConv2dS6  = AttentionSubspaceBlueprintSeparableConv2d6
ABSConv2dS7  = AttentionSubspaceBlueprintSeparableConv2d7
ABSConv2dS8  = AttentionSubspaceBlueprintSeparableConv2d8
ABSConv2dS9  = AttentionSubspaceBlueprintSeparableConv2d9
ABSConv2dS10 = AttentionSubspaceBlueprintSeparableConv2d10
ABSConv2dS11 = AttentionSubspaceBlueprintSeparableConv2d11
ABSConv2dS12 = AttentionSubspaceBlueprintSeparableConv2d12
ABSConv2dS13 = AttentionSubspaceBlueprintSeparableConv2d13

LAYERS.register(module=ABSConv2dS1)
LAYERS.register(module=ABSConv2dS2)
LAYERS.register(module=ABSConv2dS3)
LAYERS.register(module=ABSConv2dS4)
LAYERS.register(module=ABSConv2dS5)
LAYERS.register(module=ABSConv2dS6)
LAYERS.register(module=ABSConv2dS7)
LAYERS.register(module=ABSConv2dS8)
LAYERS.register(module=ABSConv2dS9)
LAYERS.register(module=ABSConv2dS10)
LAYERS.register(module=ABSConv2dS11)
LAYERS.register(module=ABSConv2dS12)
LAYERS.register(module=ABSConv2dS13)
