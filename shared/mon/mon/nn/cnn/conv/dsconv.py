#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements depthwise-separable convolutional layers."""

__all__ = [
    "DSConv2d",
]

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class DSConv2d(nn.Module):
    """Applies a 2D depthwise separable convolution over an input signal composed
    of several input planes.
    
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        kwargs: Additional keyword arguments for ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        *args, **kwargs
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, *args, **kwargs)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pw_conv(self.dw_conv(input))
