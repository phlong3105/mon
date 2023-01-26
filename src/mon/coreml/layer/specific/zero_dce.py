#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for Zero-DCE
models.
"""

from __future__ import annotations

__all__ = [
    "DCE", "PixelwiseHigherOrderLECurve",
]

from typing import Any

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common
from mon.coreml.typing import Int2T


@constant.LAYER.register()
class DCE(base.ConvLayerParsingMixin, nn.Module):

    def __init__(
        self,
        in_channels : int   = 3,
        out_channels: int   = 24,
        mid_channels: int   = 32,
        kernel_size : Int2T = 3,
        stride      : Int2T = 1,
        padding     : Int2T = 1,
        dilation    : Int2T = 1,
        groups      : int   = 1,
        bias        : bool  = True,
        padding_mode: str   = "zeros",
        device      : Any   = None,
        dtype       : Any   = None,
        *args, **kwargs
    ):
        super().__init__()
        self.relu  = common.ReLU(inplace=True)
        self.conv1 = common.Conv2d(
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
        self.conv2 = common.Conv2d(
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
        self.conv3 = common.Conv2d(
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
        self.conv4 = common.Conv2d(
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
        self.conv5 = common.Conv2d(
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
        self.conv6 = common.Conv2d(
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
        y1 = self.relu(self.conv1(x))
        y2 = self.relu(self.conv2(y1))
        y3 = self.relu(self.conv3(y2))
        y4 = self.relu(self.conv4(y3))
        y5 = self.relu(self.conv5(torch.cat([y3, y4],  dim=1)))
        y6 = self.relu(self.conv6(torch.cat([y2, y5],  dim=1)))
        y  = torch.tanh(self.conv7(torch.cat([y1, y6], dim=1)))
        return y


@constant.LAYER.register()
class PixelwiseHigherOrderLECurve(base.MergingLayerParsingMixin, nn.Module):
    """Pixelwise Light-Enhancement Curve is a higher-order curves that can be
    applied iteratively to enable more versatile adjustment to cope with
    challenging low-light conditions:
        LE_{n}(x) = LE_{n−1}(x) + A_{n}(x) * LE_{n−1}(x)(1 − LE_{n−1}(x)),
        
        where `A` is a parameter map with the same size as the given image, and
        `n` is the number of iteration, which controls the curvature.
    
    This module is designed to accompany both:
        - ZeroDCE   (estimate 3*n curve parameter maps)
        - ZeroDCE++ (estimate 3   curve parameter maps)
    
    Args:
        n: Number of iterations.
    """
    
    def __init__(self, n: int, *args, **kwargs):
        super().__init__()
        self.n = n
    
    def forward(
        self,
        input: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Split
        y = input[0]  # Trainable curve parameter learned from previous layer
        x = input[1]  # Original input image
        
        # Prepare curve parameter
        _, c1, _, _ = x.shape  # Should be 3
        _, c2, _, _ = y.shape  # Should be 3*n
        single_map  = True
        
        if c2 == c1 * self.n:
            single_map = False
            y = torch.split(y, c1, dim=1)
        elif c2 == 3:
            pass
        else:
            raise ValueError(
                f"Curve parameter maps `a` must be `3` or `3 * {self.n}`. "
                f"But got: {c2}."
            )
        
        # Estimate curve parameter
        for i in range(self.n):
            y_i = y if single_map else y[i]
            x   = x + y_i * (torch.pow(x, 2) - x)

        y = list(y)             if isinstance(y, tuple) else y
        y = torch.cat(y, dim=1) if isinstance(y, list)  else y
        return y, x
