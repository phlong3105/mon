#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise-separable convolutional layers.

This module implements the depthwise separable convolutional layers used for
lightweight feature extraction.
"""

from __future__ import annotations

__all__ = [
    "DSConv2d",
]

from typing import Any

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


# ==============================================================================
# region LAYERS
# ==============================================================================

class DSConv2d(nn.Module):
    """Depthwise separable convolutional layer.
    
    This layer consists of a depthwise convolution followed by a pointwise
    convolution.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        """Initialize a new instance.
        
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Defaults to 1.
            padding: Padding added to both sides of the input. Defaults to 0.
            dilation: Spacing between kernel elements. Defaults to 1.
            groups: Number of blocked connections from input channels to output
                channels. Defaults to 1.
            bias: If True, adds a learnable bias to the output. Defaults to True.
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
                Defaults to 'zeros'.
            device: The device on which to allocate the parameters.
            dtype: The data type of the parameters.
        """
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = False,  # Bias is redundant in depthwise conv if followed by pointwise
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = groups,
            bias         = bias,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (B, C_in, H, W) and values
                ranging from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (B, C_out, H_out, W_out) and values
                ranging from 0.0 to 1.0.
        """
        return self.pw_conv(self.dw_conv(input))

# endregion
