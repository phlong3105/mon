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

from mon import core
from mon.coreml import constant
from mon.coreml.layer import base
from mon.coreml.typing import Int2T, IntAnyT


# region Helper Function

def get_same_padding(
    x          : int,
    kernel_size: int,
    stride     : int,
    dilation   : int
) -> int:
    """Calculate asymmetric TensorFlow-like 'same' padding value for 1
    dimension of the convolution.
    """
    return max((core.math.ceil(x / stride) - 1) * stride +
               (kernel_size - 1) * dilation + 1 - x, 0)


def get_symmetric_padding(
    kernel_size: int,
    stride     : int = 1,
    dilation   : int = 1,
    *args, **kwargs
) -> int:
    """Calculate symmetric padding for a convolution."""
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def to_same_padding(
    kernel_size: IntAnyT,
    padding    : IntAnyT | None = None,
    *args, **kwargs
) -> int | list | None:
    """It takes a kernel size and a padding, and if the padding is None, it
    returns None, otherwise it returns the kernel size divided by 2.
    
    Args:
        kernel_size: The size of the convolutional kernel.
        padding: The padding to use for the convolution.
    
    Returns:
        The padding is being returned.
    """
    if padding is None:
        if isinstance(kernel_size, int):
            return kernel_size // 2
        if isinstance(kernel_size, (tuple, list)):
            return [k // 2 for k in kernel_size]
    return padding


def pad_same(
    input      : torch.Tensor,
    kernel_size: Int2T,
    stride     : Int2T,
    dilation   : Int2T = (1, 1),
    value      : float = 0,
    *args, **kwargs
):
    """Dynamically pad input tensor with 'same' padding for conv with specified
    args.
    """
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

@constant.LAYER.register()
class ConstantPad1d(base.PassThroughLayerParsingMixin, nn.ConstantPad1d):
    pass


@constant.LAYER.register()
class ConstantPad2d(base.PassThroughLayerParsingMixin, nn.ConstantPad2d):
    pass


@constant.LAYER.register()
class ConstantPad3d(base.PassThroughLayerParsingMixin, nn.ConstantPad3d):
    pass


@constant.LAYER.register()
class ZeroPad2d(base.PassThroughLayerParsingMixin, nn.ZeroPad2d):
    pass

# endregion


# region Reflection Padding

@constant.LAYER.register()
class ReflectionPad1d(base.PassThroughLayerParsingMixin, nn.ReflectionPad1d):
    pass


@constant.LAYER.register()
class ReflectionPad2d(base.PassThroughLayerParsingMixin, nn.ReflectionPad2d):
    pass


@constant.LAYER.register()
class ReflectionPad3d(base.PassThroughLayerParsingMixin, nn.ReflectionPad3d):
    pass

# endregion


# region Replication Padding

@constant.LAYER.register()
class ReplicationPad1d(base.PassThroughLayerParsingMixin, nn.ReplicationPad1d):
    pass


@constant.LAYER.register()
class ReplicationPad2d(base.PassThroughLayerParsingMixin, nn.ReplicationPad2d):
    pass


@constant.LAYER.register()
class ReplicationPad3d(base.PassThroughLayerParsingMixin, nn.ReplicationPad3d):
    pass

# endregion
