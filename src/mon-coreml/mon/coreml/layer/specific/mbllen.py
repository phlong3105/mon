#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for MBLLEN models.
"""

from __future__ import annotations

__all__ = [
    "EM", "EnhancementModule",
]

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import Int2T


@constant.LAYER.register()
class EnhancementModule(base.PassThroughLayerParsingMixin, nn.Module):
    """Enhancement regression (EM) has a symmetric structure to first apply
    convolutions and then deconvolutions.
    
    Args:
        in_channels: Number of input channels. Defaults to 32.
        mid_channels: Number of input and output channels for middle Conv2d
            layers used in each EM block. Defaults to 8.
        out_channels: Number of output channels. Defaults to 3.
        kernel_size: Kernel size for Conv2d layers used in each EM block.
            Defaults to 5.
    """
    
    def __init__(
        self,
        in_channels : int   = 32,
        mid_channels: int   = 8,
        out_channels: int   = 3,
        kernel_size : Int2T = 5,
        *args, **kwargs
    ):
        super().__init__()
        self.convs = torch.nn.Sequential(
            common.Conv2d(
                in_channels  = in_channels,
                out_channels = mid_channels,
                kernel_size  = 3,
                padding      = 1,
                padding_mode = "replicate"
            ),
            common.ReLU(),
            common.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ),
            common.ReLU(),
            common.Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels * 2,
                kernel_size  = kernel_size
            ),
            common.ReLU(),
            common.Conv2d(
                in_channels  = mid_channels * 2,
                out_channels = mid_channels * 4,
                kernel_size  = kernel_size
            ),
            common.ReLU()
        )
        self.deconvs = torch.nn.Sequential(
            common.ConvTranspose2d(
                in_channels  = mid_channels * 4,
                out_channels = mid_channels * 2,
                kernel_size  = kernel_size,
            ),
            common.ReLU(),
            common.ConvTranspose2d(
                in_channels  = mid_channels * 2,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ),
            common.ReLU(),
            common.Conv2d(
                in_channels  = mid_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size
            ),
            common.ReLU(),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.deconvs(self.convs(x))
        return y


EM = EnhancementModule
constant.LAYER.register(module=EM)
