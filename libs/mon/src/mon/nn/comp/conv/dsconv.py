#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise-separable convolutional layers.

This module implements the depthwise separable convolutional layers used for
lightweight feature extraction.
"""

__all__ = [
    "DSConv2d",
]

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class DSConv2d(nn.Module):
    """Depthwise separable convolutional layer."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            *args: Additional positional arguments for the convolutional layers.
            **kwargs: Additional keyword arguments for the convolutional layers.
        """
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, *args, **kwargs)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, *args, **kwargs)

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
