#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for MobileOne
models.
"""

from __future__ import annotations

__all__ = [
    "MobileOneStage",
]

import torch
from torch import nn

from mon.coreml.layer import base, common
from mon.globals import LAYERS


@LAYERS.register()
class MobileOneStage(base.ConvLayerParsingMixin, nn.Module):
    """MobileOneStage used to construct the MobileOne Model from the paper:
    "An Improved One millisecond Mobile Backbone" (https://arxiv.org/pdf/2206.04040.pdf).
    
    References:
        https://github.com/apple/ml-mobileone/blob/main/mobileone.py
    """
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        num_blocks       : int,
        num_se_blocks    : int,
        inference_mode   : bool = False,
        num_conv_branches: int  = 1,
    ):
        super().__init__()
        strides = [2] + [1] * (num_blocks - 1)
        convs   = []
        for ix, stride in enumerate(strides):
            se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    f"Require number of SE blocks less than number of layers. "
                    f"But got: {num_se_blocks} > {num_blocks}."
                )
            if ix >= (num_blocks - num_se_blocks):
                se = True
            
            # Depthwise
            convs.append(
                common.MobileOneConv2d(
                    in_channels       = in_channels,
                    out_channels      = in_channels,
                    kernel_size       = 3,
                    stride            = stride,
                    padding           = 1,
                    groups            = in_channels,
                    inference_mode    = inference_mode,
                    se                = se,
                    num_conv_branches = num_conv_branches,
                )
            )
            # Pointwise
            convs.append(
                common.MobileOneConv2d(
                    in_channels       = in_channels,
                    out_channels      = out_channels,
                    kernel_size       = 1,
                    stride            = 1,
                    padding           = 0,
                    groups            = 1,
                    inference_mode    = inference_mode,
                    se                = se,
                    num_conv_branches = num_conv_branches,
                )
            )
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)
    
    def reparameterize(self):
        for module in self.convs.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.convs(x)
        return y
