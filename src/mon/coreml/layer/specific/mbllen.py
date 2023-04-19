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

from mon.coreml.layer import base, common
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


@LAYERS.register()
class EnhancementModule(base.PassThroughLayerParsingMixin, nn.Module):
    """Enhancement regression (EM) has a symmetric structure to first apply
    convolutions and then deconvolutions.
    """
    
    def __init__(
        self,
        in_channels : int       = 32,
        mid_channels: int       = 8,
        out_channels: int       = 3,
        kernel_size : _size_2_t = 5,
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
LAYERS.register(module=EM)
