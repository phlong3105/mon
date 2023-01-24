#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for VDSR models.
"""

from __future__ import annotations

__all__ = [
    "VDSR",
]

from typing import Any

import torch
from torch import nn

from mon.core import math
from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import Int2T


@constant.LAYER.register()
class VDSR(base.ConvLayerParsingMixin, nn.Module):
    """VDSR (Very Deep Super-Resolution).
    
    References:
        https://cv.snu.ac.kr/research/VDSR/
        https://github.com/twtygqyy/pytorch-vdsr
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T       = 3,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 1,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = False,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = common.Conv2d(
            in_channels  = in_channels,
            out_channels = 64,
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
        self.residual_layer = nn.Sequential(*[
            common.Conv2dNormActivation(
                in_channels      = 64,
                out_channels     = 64,
                kernel_size      = kernel_size,
                stride           = stride,
                padding          = padding,
                dilation         = dilation,
                groups           = groups,
                bias             = bias,
                norm_layer       = None,
                activation_layer = common.ReLU,
            )
            for _ in range(18)
        ])
        self.conv2 = common.Conv2d(
            in_channels  = 64,
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
        self.relu = common.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, common.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.relu(self.conv1(x))
        y = self.residual_layer(y)
        y = self.conv2(y)
        y = torch.add(y, x)
        return y
