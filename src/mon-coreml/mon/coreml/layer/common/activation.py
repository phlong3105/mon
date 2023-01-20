#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements activation layers."""

from __future__ import annotations

__all__ = [
    "ArgMax", "CELU", "Clamp", "Clip", "ELU", "FReLU", "GELU", "GLU",
    "Hardshrink", "Hardsigmoid", "Hardswish", "Hardtanh", "LeakyReLU",
    "LogSigmoid", "LogSoftmax", "Mish", "MultiheadAttention", "PReLU", "ReLU",
    "ReLU6", "RReLU", "SELU", "Sigmoid", "SiLU", "Softmax", "Softmax2d",
    "Softmin", "Softplus", "Softshrink", "Softsign", "Tanh", "Tanhshrink",
    "Threshold", "hard_sigmoid", "to_act_layer",
]

import functools
import types
from typing import Callable

import torch
from torch import nn
from torch.nn import functional

from mon import foundation
from mon.coreml import constant
from mon.coreml.layer import base
from mon.coreml.typing import CallableType, Int2T


# region Linear Unit

@constant.LAYER.register()
class CELU(base.PassThroughLayerParsingMixin, nn.CELU):
    pass


@constant.LAYER.register()
class ELU(base.PassThroughLayerParsingMixin, nn.ELU):
    pass


class FReLU(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        c1: int,
        k : Int2T = 3,
        *args, **kwargs
    ):
        super().__init__()
        k         = foundation.to_2tuple(k)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.act  = nn.BatchNorm2d(c1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = max(x, self.act(self.conv(x)))
        return y


@constant.LAYER.register()
class GELU(base.PassThroughLayerParsingMixin, nn.GELU):
    pass


@constant.LAYER.register()
class GLU(base.PassThroughLayerParsingMixin, nn.GLU):
    pass


@constant.LAYER.register()
class LeakyReLU(base.PassThroughLayerParsingMixin, nn.LeakyReLU):
    pass


@constant.LAYER.register()
class PReLU(base.PassThroughLayerParsingMixin, nn.PReLU):
    pass


@constant.LAYER.register()
class ReLU(base.PassThroughLayerParsingMixin, nn.ReLU):
    pass


@constant.LAYER.register()
class ReLU6(base.PassThroughLayerParsingMixin, nn.ReLU6):
    pass


@constant.LAYER.register()
class RReLU(base.PassThroughLayerParsingMixin, nn.RReLU):
    pass


@constant.LAYER.register()
class SELU(base.PassThroughLayerParsingMixin, nn.SELU):
    pass


@constant.LAYER.register()
class SiLU(base.PassThroughLayerParsingMixin, nn.SiLU):
    pass

# endregion


# region Shrink

class Hardshrink(base.PassThroughLayerParsingMixin, nn.Hardshrink):
    pass


class Softshrink(base.PassThroughLayerParsingMixin, nn.Softshrink):
    pass


class Tanhshrink(base.PassThroughLayerParsingMixin, nn.Tanhshrink):
    pass

# endregion


# region Sigmoid

def hard_sigmoid(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return input.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return functional.relu6(input + 3.0) / 6.0


class Hardsigmoid(base.PassThroughLayerParsingMixin, nn.Hardsigmoid):
    pass


class LogSigmoid(base.PassThroughLayerParsingMixin, nn.LogSigmoid):
    pass


class Sigmoid(base.PassThroughLayerParsingMixin, nn.Sigmoid):
    pass

# endregion


# region Softmax

class LogSoftmax(base.PassThroughLayerParsingMixin, nn.LogSoftmax):
    pass


class Softmax(base.PassThroughLayerParsingMixin, nn.Softmax):
    pass


class Softmax2d(base.PassThroughLayerParsingMixin, nn.Softmax2d):
    pass

# endregion


# region Tanh

class Hardtanh(base.PassThroughLayerParsingMixin, nn.Hardtanh):
    pass


class Tanh(base.PassThroughLayerParsingMixin, nn.Tanh):
    pass

# endregion


# region Misc

class ArgMax(base.PassThroughLayerParsingMixin, nn.Module):
    """Finds indices of maximum values of a tensor along a given dimension :param`dim`.
    
    Args:
        dim: A dimension to find indices of maximum values. Defaults to None.
    """
    
    def __init__(self, dim: int | None = None, *args, **kwargs):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.argmax(x, dim=self.dim)
        return y


class Clamp(base.PassThroughLayerParsingMixin, nn.Module):
    """Clamps a tensor' values within a range of [min, max].

    Args:
        min: The lower-bound of the range to be clamped to. Defaults to -1.0.
        max: The upper-bound of the range to be clamped to. Defaults to -1.0.
    """
    
    def __init__(
        self,
        min: float = -1.0,
        max: float =  1.0,
        *args, **kwargs
    ):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.clamp(x, min=self.min, max=self.max)
        return y


Clip = Clamp


class Hardswish(base.PassThroughLayerParsingMixin, nn.Hardswish):
    pass


class Mish(base.PassThroughLayerParsingMixin, nn.Mish):
    pass


class MultiheadAttention(
    base.PassThroughLayerParsingMixin,
    nn.MultiheadAttention
):
    pass


class Softmin(base.PassThroughLayerParsingMixin, nn.Softmin):
    pass


class Softplus(base.PassThroughLayerParsingMixin, nn.Softplus):
    pass


class Softsign(base.PassThroughLayerParsingMixin, nn.Softsign):
    pass


class Threshold(base.PassThroughLayerParsingMixin, nn.Threshold):
    pass

# endregion


def to_act_layer(
    act    : CallableType | None = ReLU(),
    inplace: bool                = True,
    *args, **kwargs
) -> nn.Module:
    """Create activation layer."""
    # if isinstance(act, str):
    #     act = LAYER.build(name=act)
    act_layer = act
    if act is None:
        act_layer = nn.Identity()
    elif isinstance(act, Callable | types.FunctionType | functools.partial):
        act_layer = act(*args, **kwargs)
    return act_layer
