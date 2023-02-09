#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for HINet models.
"""

from __future__ import annotations

__all__ = [
    "HINetConvBlock", "HINetSkipBlock", "HINetUpBlock",
]

from typing import Sequence

import torch
from torch import nn

from mon.coreml.layer import base, common
from mon.globals import LAYERS


@LAYERS.register()
class HINetConvBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool  = False,
        use_hin     : bool  = False,
    ):
        super().__init__()
        self.downsample = downsample
        self.use_csff   = use_csff
        self.use_hin    = use_hin
        
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu1 = common.LeakyReLU(relu_slope, inplace=False)
        self.conv2 = common.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu2    = common.LeakyReLU(relu_slope, inplace = False)
        self.identity = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        
        if downsample and use_csff:
            self.csff_enc = common.Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
            self.csff_dec = common.Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
        
        if self.use_hin:
            self.norm = common.InstanceNorm2d(out_channels // 2, affine=True)
        
        if downsample:
            self.downsample = common.Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1,
                bias         = False,
            )
    
    def forward(
        self,
        input: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        
        Args:
            input: A single tensor for the first UNet or a list of 3 tensors for
                the second UNet.

        Returns:
            Output tensors.
        """
        enc = dec = None
        if isinstance(input, torch.Tensor):
            x = input
        elif isinstance(input, Sequence):
            x = input[0]  # Input
            if len(input) == 2:
                enc = input[1]  # Encode path
            if len(input) == 3:
                dec = input[2]  # Decode path
        else:
            raise TypeError()
        
        y = self.conv1(x)
        if self.use_hin:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y      = torch.cat([self.norm(y1), y2], dim=1)
        y  = self.relu1(y)
        y  = self.relu2(self.conv2(y))
        y += self.identity(x)
        
        if enc is not None and dec is not None:
            if not self.use_csff:
                raise ValueError()
            y = y + self.csff_enc(enc) + self.csff_dec(dec)
        
        if self.downsample:
            y_down = self.downsample(y)
            return y_down, y
        else:
            return None, y


@LAYERS.register()
class HINetUpBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
    ):
        super().__init__()
        self.up = common.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv = HINetConvBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            downsample   = False,
            relu_slope   = relu_slope,
        )
    
    def forward(
        self,
        input: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        assert isinstance(input, Sequence) and len(input) == 2
        x    = input[0]
        skip = input[1]
        x_up = self.up(x)
        y    = torch.cat([x_up, skip], dim=1)
        y    = self.conv(y)
        y    = y[-1]
        return y


@LAYERS.register()
class HINetSkipBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        mid_channels: int = 128,
        repeat_num  : int = 1,
    ):
        super().__init__()
        self.repeat_num = repeat_num
        self.shortcut   = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            bias         = True
        )
        blocks = []
        blocks.append(
            HINetConvBlock(
                in_channels  = in_channels,
                out_channels = mid_channels,
                downsample   = False,
                relu_slope   = 0.2
            )
        )
        for i in range(self.repeat_num - 2):
            blocks.append(
                HINetConvBlock(
                    in_channels  = mid_channels,
                    out_channels = mid_channels,
                    downsample   = False,
                    relu_slope   = 0.2
                )
            )
        blocks.append(
            HINetConvBlock(
                in_channels  = mid_channels,
                out_channels = out_channels,
                downsample   = False,
                relu_slope   = 0.2
            )
        )
        self.blocks = torch.nn.Sequential(*blocks)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x_skip = self.shortcut(x)
        y      = self.blocks(x)
        y      = y + x_skip
        return y
