#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for depthwise-separable convolutional layers.

This module provides a depthwise separable convolutional layer for 2D inputs.
"""

__all__ = [
    "DSConv2d",
]

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class DSConv2d(nn.Module):
    """A depthwise separable 2D convolutional layer.
    
    It applies a 2D depthwise separable convolution over an input signal
    composed of several input planes.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        *args, **kwargs
    ):
        """Initializes the DSConv2d layer.
        
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (_size_2_t): Size of the convolving kernel.
            *args: Additional positional arguments for the convolutional layers.
            **kwargs: Additional keyword arguments for the convolutional layers.
        """
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, *args, **kwargs)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DSConv2d layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (N, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        return self.pw_conv(self.dw_conv(input))
