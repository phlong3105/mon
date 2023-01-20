#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and common. especially used for FINet models.
"""

from __future__ import annotations

__all__ = [
    "FINetConvBlock", "FINetGhostConv", "FINetGhostUpBlock", "FINetUpBlock",
]

from typing import Sequence

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import CallableType


@constant.LAYER.register()
class FINetConvBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool               = False,
        use_norm    : bool               = False,
        p           : float              = 0.5,
        scheme      : str                = "half",
        pool        : CallableType | str = "avg",
        bias        : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.downsample = downsample
        self.use_csff   = use_csff
        self.use_norm   = use_norm
        self.p          = p
        
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        # self.relu1 = LeakyReLU(relu_slope, inplace=False)
        self.relu1 = common.GELU()
        self.conv2 = common.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        # self.relu2    = LeakyReLU(relu_slope, inplace=False)
        self.relu2    = common.GELU()
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
        
        if self.use_norm:
            self.norm = common.FractionalInstanceNorm2d(
                num_features = out_channels,
                p            = self.p,
                scheme       = scheme,
                pool         = pool,
                bias         = bias,
            )

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
        input: torch.Tensor | Sequence[torch.Tensor]
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
        
        y  = self.conv1(x)
        if self.use_norm:
            y = self.norm(y)
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


@constant.LAYER.register()
class FINetUpBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
        use_norm    : bool               = False,
        p           : float              = 0.5,
        scheme      : str                = "half",
        pool        : CallableType | str = "avg",
        bias        : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.up = common.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv = FINetConvBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            downsample   = False,
            relu_slope   = relu_slope,
            use_norm     = use_norm,
            p            = p,
            scheme       = scheme,
            pool         = pool,
            bias         = bias,
            *args, **kwargs
        )
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        assert isinstance(input, Sequence) and len(input) == 2
        x    = input[0]
        skip = input[1]
        x_up = self.up(x)
        y    = torch.cat([x_up, skip], dim=1)
        y    = self.conv(y)
        y    = y[-1]
        return y


@constant.LAYER.register()
class FINetGhostConv(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool               = False,
        use_norm    : bool               = False,
        p           : float              = 0.5,
        scheme      : str                = "half",
        pool        : CallableType | str = "avg",
        bias        : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.downsample = downsample
        self.use_csff   = use_csff
        self.use_norm   = use_norm
        self.p          = p
        
        self.conv1 = common.GhostConv2d(
            in_channels    = in_channels,
            out_channels   = out_channels,
            dw_kernel_size = 3,
            stride         = 1,
            bias           = True,
        )
        self.relu1 = common.GELU()
        self.conv2 = common.GhostConv2d(
            in_channels    = out_channels,
            out_channels   = out_channels,
            dw_kernel_size = 3,
            stride         = 1,
            bias           = True,
        )
        self.relu2    = common.GELU()
        self.identity = common.GhostConv2d(
            in_channels    = in_channels,
            out_channels   = out_channels,
            dw_kernel_size = 1,
            stride         = 1,
            padding        = 0,
        )
        
        if downsample and use_csff:
            self.csff_enc = common.GhostConv2d(
                in_channels    = out_channels,
                out_channels   = out_channels,
                dw_kernel_size = 3,
                stride         = 1,
                padding        = 1,
            )
            self.csff_dec = common.GhostConv2d(
                in_channels    = out_channels,
                out_channels   = out_channels,
                dw_kernel_size = 3,
                stride         = 1,
                padding        = 1,
            )
        
        if self.use_norm:
            self.norm = common.FractionalInstanceNorm2d(
                num_features = out_channels,
                p            = self.p,
                scheme       = scheme,
                pool         = pool,
                bias         = bias,
            )

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
        input: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        
        Args:
            input (Tensors): A single tensor for the first UNet or a list of 3
                tensors for the second UNet.

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
        
        y  = self.conv1(x)
        if self.use_norm:
            y = self.norm(y)
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


@constant.LAYER.register()
class FINetGhostUpBlock(base.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
        use_norm    : bool               = False,
        p           : float              = 0.5,
        scheme      : str                = "half",
        pool        : CallableType | str = "avg",
        bias        : bool               = True,
        *args, **kwargs
    ):
        super().__init__()
        self.up = common.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv = FINetGhostConv(
            in_channels  = in_channels,
            out_channels = out_channels,
            downsample   = False,
            relu_slope   = relu_slope,
            use_norm     = use_norm,
            p            = p,
            scheme       = scheme,
            pool         = pool,
            bias         = bias,
            *args, **kwargs
        )
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        assert isinstance(input, Sequence) and len(input) == 2
        x    = input[0]
        skip = input[1]
        x_up = self.up(x)
        y    = torch.cat([x_up, skip], dim=1)
        y    = self.conv(y)
        y    = y[-1]
        return y
