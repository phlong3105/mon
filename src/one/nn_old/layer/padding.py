#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Padding Layers.
"""

from __future__ import annotations

import math
from typing import Union

import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import Int2T
from one.core import PADDING_LAYERS

__all__ = [
    "autopad", 
    "get_padding",
    "get_padding_value",
    "get_same_padding",
    "is_static_pad", 
    "pad_same"
]


# MARK: - Functional

def autopad(kernel_size: Int2T, padding: Union[str, Int2T, None] = None):
    """Pad to `same`."""
    if padding is None:
        padding = (kernel_size // 2 if isinstance(kernel_size, int)
                   else [input // 2 for input in kernel_size])  # auto-pad
    return padding


def pad_same(
    x          : Tensor,
    kernel_size: Int2T,
    stride     : Int2T,
    dilation   : Int2T = (1, 1),
    value      : float  = 0
):
    """Dynamically pad input with 'same' padding for conv with specified
    args.
    """
    ih, iw = x.size()[-2:]
    pad_h  = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w  = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(
            x,
            [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            value=value
        )
    return x


def get_padding_value(
    padding: Union[str, Int2T, None], kernel_size: Int2T, **kwargs
) -> tuple[(tuple, int), bool]:
    dynamic = False
    if isinstance(padding, str):
        # For any string padding, the padding will be calculated for you, one
        # of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory
            # allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # Dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    """Calculate symmetric padding for a convolution.

    FYI: `**_` mean ignore the rest of the args.
    """
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int) -> int:
    """Calculate asymmetric TensorFlow-like 'same' padding for a convolution.
    """
    return max((math.ceil(x / stride) - 1) * stride +
               (kernel_size - 1) * dilation + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> bool:
    """Can `same` padding for given args be done statically?."""
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# MARK: - Register

PADDING_LAYERS.register(name="zero",        module=nn.ZeroPad2d)
PADDING_LAYERS.register(name="reflection",  module=nn.ReflectionPad2d)
PADDING_LAYERS.register(name="replication", module=nn.ReplicationPad2d)
