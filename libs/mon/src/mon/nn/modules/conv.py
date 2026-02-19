#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convolutional Layers.

This module contains various convolutional layers - the basic building blocks
of convolutional neural networks.
"""

from __future__ import annotations

__all__ = [
    "DSConv2d",
]

from torch import nn, Tensor
from torch.nn.common_types import _size_2_t


# ==============================================================================
# region LAYERS
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
