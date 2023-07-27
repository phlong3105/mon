#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements padding layers."""

from __future__ import annotations

__all__ = [
    "ConstantPad1d", "ConstantPad2d", "ConstantPad3d", "get_same_padding",
    "get_symmetric_padding", "pad_same", "ReflectionPad1d", "ReflectionPad2d",
    "ReflectionPad3d", "ReplicationPad1d", "ReplicationPad2d",
    "ReplicationPad3d", "to_same_padding", "ZeroPad2d",
]

import torch
from torch import nn
from torch.nn import functional

from mon.core import math
from mon.globals import LAYERS
from mon.nn.layer import base
from mon.nn.typing import _size_2_t


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
    padding    : _size_2_t | None = None,
) -> int | list | None:
    """It takes a kernel size and a padding, and if the padding is ``None``, it
    returns ``None``, otherwise it returns the kernel size divided by ``2``.
    
    Args:
        kernel_size: The size of the convolutional kernel.
        padding: The padding to use for the convolution.
    
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
        x = functional.pad(
            input = x,
            pad   = [pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2],
            value = value
        )
    return x


# endregion


# region Constant Padding

@LAYERS.register()
class ConstantPad1d(base.PassThroughLayerParsingMixin, nn.ConstantPad1d):
    pass


@LAYERS.register()
class ConstantPad2d(base.PassThroughLayerParsingMixin, nn.ConstantPad2d):
    pass


@LAYERS.register()
class ConstantPad3d(base.PassThroughLayerParsingMixin, nn.ConstantPad3d):
    pass


@LAYERS.register()
class ZeroPad2d(base.PassThroughLayerParsingMixin, nn.ZeroPad2d):
    pass


# endregion


# region Reflection Padding

@LAYERS.register()
class ReflectionPad1d(base.PassThroughLayerParsingMixin, nn.ReflectionPad1d):
    pass


@LAYERS.register()
class ReflectionPad2d(base.PassThroughLayerParsingMixin, nn.ReflectionPad2d):
    pass


@LAYERS.register()
class ReflectionPad3d(base.PassThroughLayerParsingMixin, nn.ReflectionPad3d):
    pass


# endregion


# region Replication Padding

@LAYERS.register()
class ReplicationPad1d(base.PassThroughLayerParsingMixin, nn.ReplicationPad1d):
    pass


@LAYERS.register()
class ReplicationPad2d(base.PassThroughLayerParsingMixin, nn.ReplicationPad2d):
    pass


@LAYERS.register()
class ReplicationPad3d(base.PassThroughLayerParsingMixin, nn.ReplicationPad3d):
    pass

# endregion
