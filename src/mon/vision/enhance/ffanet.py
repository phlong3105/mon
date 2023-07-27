#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements FFANet models."""

from __future__ import annotations

__all__ = [
    "FFA", "FFABlock", "FFAGroup", "FFANet", "FFAPostProcess", "FFAPreProcess",
]

from typing import Any, Sequence

import torch

from mon import nn
from mon.core import builtins, pathlib
from mon.globals import LAYERS, MODELS
from mon.nn import _size_2_t
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class FFA(nn.SameChannelsLayerParsingMixin, nn.Module):
    """This is the main feature in FFA-Net, the Feature Fusion Attention.
    
    We concatenate all feature maps output by G Group Architectures in the
    channel direction. Furthermore, We fuse features by multiplying the adaptive
    learning weights which are obtained by Feature Attention (FA) mechanism.
    
    Args:
        num_groups: Number of groups used in FFA-Net.
    """

    def __init__( self, channels: int, num_groups: int):
        super().__init__()
        self.channels   = channels
        self.num_groups = num_groups
        self.ca = nn.Sequential(
            *[
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    in_channels  = self.channels * self.num_groups,
                    out_channels = self.channels // 16,
                    kernel_size  = 1,
                    padding      = 0,
                    bias         = True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels  = self.channels // 16,
                    out_channels = self.channels * self.num_groups,
                    kernel_size  = 1,
                    padding      = 0,
                    bias         = True
                ),
                nn.Sigmoid()
            ]
        )
        self.pa = nn.PixelAttentionModule(
            channels        = self.channels,
            reduction_ratio = 8,
            kernel_size     = 1,
        )
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        x = input
        assert isinstance(x, list | tuple) and len(x) == self.num_groups
        w = self.ca(torch.cat(builtins.to_list(x), dim=1))
        w = w.view(-1, self.num_groups, self.channels)[:, :, :, None, None]
        y = w[:, 0, ::] * x[0]
        for i in range(1, len(x)):
            y += w[:, i, ::] * x[i]
        return y


@LAYERS.register()
class FFABlock(nn.SameChannelsLayerParsingMixin, nn.Module):
    """A basic block structure in FFA-Net."""

    def __init__(self, channels: int, kernel_size: _size_2_t):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.ca = nn.ChannelAttentionModule(
            channels        = channels,
            reduction_ratio = 8,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = True,
            max_pool        = False,
        )
        self.pa = nn.PixelAttentionModule(
            channels        = channels,
            reduction_ratio = 8,
            kernel_size     = 1,
            stride          = 1,
            padding         = 0,
            bias            = True,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.act1(self.conv1(x))
        y = y + x
        y = self.conv2(y)
        y = self.ca(y)
        y = self.pa(y)
        y += x
        return y


@LAYERS.register()
class FFAGroup(nn.SameChannelsLayerParsingMixin, nn.Module):
    """Our Group Architecture combines B Basic Block structures with skip
    connections module. Continuous B blocks increase the depth and
    expressiveness of the FFA-Net. And skip connections make FFA-Net get around
    training difficulty. At the end of the FFA-Net, we add a recovery part using
    a two-layer convolutional network implementation and a long shortcut global
    residual learning module. Finally, we restore our desired haze-free image.
    """
    
    def __init__(
        self,
        channels   : int,
        kernel_size: _size_2_t,
        num_blocks : int,
    ):
        super().__init__()
        m: list[nn.Module] = [
            FFABlock(channels=channels, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ]
        m.append(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels,
                kernel_size  = kernel_size,
                padding      = (kernel_size // 2),
                bias         = True
            )
        )
        self.gp = torch.nn.Sequential(*m)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y  = self.gp(x)
        y += x
        return y


@LAYERS.register()
class FFAPostProcess(nn.ConvLayerParsingMixin, nn.Module):
    """Post-process module in FFA-Net."""
    
    def __init__(
        self,
        in_channels : int       = 64,
        out_channels: int       = 3,
        kernel_size : _size_2_t = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.conv2 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv2(self.conv1(x))
        return y


@LAYERS.register()
class FFAPreProcess(nn.ConvLayerParsingMixin, nn.Module):
    """Pre-process module in FFA-Net."""
    
    def __init__(
        self,
        in_channels : int       = 3,
        out_channels: int       = 64,
        kernel_size : _size_2_t = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = 1,
            padding      = (kernel_size // 2),
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv(x)
        return y

# endregion


# region Model

@MODELS.register(name="ffanet")
class FFANet(base.ImageEnhancementModel):
    """Half-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(self, config: Any = "ffanet.yaml", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
    
# endregion
