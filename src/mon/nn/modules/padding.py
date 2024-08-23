#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Padding Layers.

This module implements padding layers.
"""

from __future__ import annotations

__all__ = [
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
    "get_same_padding",
    "get_symmetric_padding",
    "pad_same",
    "to_same_padding",
]

import math

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.padding import *


# region Helper Function

def get_same_padding(
    x          : int,
    kernel_size: int,
    stride     : int,
    dilation   : int
) -> int:
    """Calculate asymmetric TensorFlow-like ``same`` padding value for 1
    dimension of the convolution.
    """
    return max(
        (math.ceil(x / stride) - 1) * stride
        + (kernel_size - 1) * dilation + 1 - x, 0
    )


def get_symmetric_padding(
    kernel_size: int,
    stride     : int = 1,
    dilation   : int = 1,
) -> int:
    """Calculate symmetric padding for a convolution."""
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def to_same_padding(
    kernel_size: _size_2_t,
    padding    : _size_2_t = None,
) -> int | list | None:
    """It takes a kernel size and a padding, and if the padding is ``None``, it
    returns ``None``, otherwise it returns the kernel size divided by ``2``.
    
    Args:
        kernel_size: The size of the convolutional kernel.
        padding: The padding to use for the convolution. Default: ``None``.
    
    Returns:
        The padding is being returned.
    """
    if padding is None:
        if isinstance(kernel_size, int):
            return kernel_size // 2
        if isinstance(kernel_size, tuple | list):
            return [k // 2 for k in kernel_size]
    return padding


def pad_same(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    stride     : _size_2_t,
    dilation   : _size_2_t = (1, 1),
    value      : float     = 0,
):
    """Pad input tensor with ``same`` padding for conv with specified args."""
    x      = input
    ih, iw = x.size()[-2:]
    pad_h  = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w  = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            input = x,
            pad   = [pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2],
            value = value
        )
    return x

# endregion
