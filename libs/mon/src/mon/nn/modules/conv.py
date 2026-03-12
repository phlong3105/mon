#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolutional Layers.

This module contains various convolutional layers - the basic building blocks
of convolutional neural networks.
"""

from __future__ import annotations

__all__ = [
    "DSConv2d",
    "Conv2dTime",
]

import torch
from torch import nn, Tensor
from torch.nn.common_types import _size_2_t


# ==============================================================================
# region LAYERS
# ==============================================================================

# --- Depthwise Separable Convolution ---

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
            kernel_size (_size_2_t): Size of the convolving kernel.
            stride (_size_2_t, optional): Stride of the convolution. Defaults to 1.
            padding (_size_2_t | str, optional): Padding added to both sides of
                the input. Defaults to 0.
            dilation (_size_2_t, optional): Spacing between kernel elements.
                Defaults to 1.
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Defaults to 1.
            bias (bool, optional): If True, adds a learnable bias to the output.
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
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H_in, W_in) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C_out, H_out, W_out) and values
                ranging from 0.0 to 1.0.
        """
        return self.pw_conv(self.dw_conv(x))


# --- Time-Dependent Convolution ---

class Conv2dTime(nn.Conv2d):
    """2D convolutional layer that takes in the time step as an additional input.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, *args, **kwargs):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of channels in the input image (excluding
                the time channel).
        """
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        t_img = torch.ones_like(x[:, :1, :, :]) * t  # (B, 1, H, W)
        t_and_x = torch.cat([t_img, x], 1)  # (B, C + 1, H, W)
        return super(Conv2dTime, self).forward(t_and_x)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
