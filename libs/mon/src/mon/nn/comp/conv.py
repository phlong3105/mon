#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolutional layers.

This module contains various convolutional layers - the basic building blocks
of convolutional neural networks.
"""

from __future__ import annotations

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DSConv2d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
]

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d,
)


# ==============================================================================
# region DEPTHWISE SEPARABLE CONVOLUTIONS
# ==============================================================================

class DSConv2d(nn.Module):
    """Depthwise separable convolutional layer.

    Apply a depthwise convolution followed by a pointwise convolution.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | tuple[int]): Size of the convolving kernel.
            stride (int | tuple[int], optional): Stride of the convolution.
                Defaults to 1.
            padding (int | tuple[int] | str): Padding added to both sides of
                the input. Defaults to 0.
            dilation (int | tuple[int]): Spacing between kernel elements.
                Defaults to 1.
            groups (int): Number of blocked connections from input channels to
                output channels. Defaults to 1.
            bias (bool): If True, adds a learnable bias to the output.
                Defaults to True.
        """
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,  # Bias is redundant in depthwise conv if followed by pointwise
            *args, **kwargs,
        )
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
            *args, **kwargs,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H_in, W_in) and
                values ranging from 0.0 to 1.0.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out) and
                values ranging from 0.0 to 1.0.
        """
        return self.pw_conv(self.dw_conv(x))


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
