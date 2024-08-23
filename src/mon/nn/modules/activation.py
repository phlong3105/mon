#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation Layers.

This module implements activation layers.
"""

from __future__ import annotations

__all__ = [
    "ArgMax",
    "CELU",
    "Clamp",
    "Clip",
    "ELU",
    "FReLU",
    "GELU",
    "GLU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "MultiheadAttention",
    "NegHardsigmoid",
    "PReLU",
    "RReLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "SiLU",
    "Sigmoid",
    "SimpleGate",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "hard_sigmoid",
    "to_act_layer",
    "xUnit",
    "xUnitD",
    "xUnitS",
]

import functools
import types
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.activation import *

from mon import core


# region Linear Unit

class FReLU(nn.Module):
    
    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        kernel_size = core.to_2tuple(kernel_size)
        self.conv   = nn.Conv2d(channels, channels, kernel_size, 1, 1, groups=channels)
        self.act    = nn.BatchNorm2d(channels)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = max(x, self.act(self.conv(x)))
        return y


class SimpleGate(nn.Module):
    """Simple gate activation unit described in the paper:
        "Simple Baselines for Image Restoration".
    
    References:
        https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x1, x2 = x.chunk(chunks=2, dim=1)
        return x1 * x2
    
# endregion


# region Sigmoid

def hard_sigmoid(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return input.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(input + 3.0) / 6.0


class NegHardsigmoid(nn.Module):
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu6(3 * input + 3.0, inplace=self.inplace) / 6.0 - 0.5

# endregion


# region xUnit

class xUnit(nn.Module):
    """xUnit activation layer.
    
    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit?ref=blog.paperspace.com
    """
    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False,
    ):
        super().__init__()
        # xUnit
        padding = (kernel_size // 2)
        self.features = nn.Sequential(
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding, groups=num_features),
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        a = self.features(x)
        y = x * a
        return y
    

class xUnitS(nn.Module):
    """Slim xUnit activation layer.
    
    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit?ref=blog.paperspace.com
    """
    
    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False,
    ):
        super().__init__()
        # slim xUnit
        padding = (kernel_size // 2)
        self.features = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding, groups=num_features),
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        a = self.features(x)
        y = x * a
        return y
    

class xUnitD(nn.Module):
    """Dense xUnit activation layer.
    
    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit?ref=blog.paperspace.com
    """
    
    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False,
    ):
        super().__init__()
        # Dense xUnit
        padding = (kernel_size // 2)
        self.features = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1, padding=0),
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size, padding=padding, groups=num_features),
            nn.BatchNorm2d(num_features) if batch_norm else nn.Identity(),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        a = self.features(x)
        y = x * a
        return y
    
# endregion


# region Misc

class ArgMax(nn.Module):
    """Finds indices of maximum values of a tensor along a given dimension
    :param`dim`.
    
    Args:
        dim: A dimension to find indices of maximum values. Default: ``None``.
    """
    
    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.argmax(x, dim=self.dim)
        return y


class Clamp(nn.Module):
    """Clamps a tensor's values within a range of `[min, max]`.

    Args:
        min: The lower-bound of the range to be clamped to. Default: ``-1.0``.
        max: The upper-bound of the range to be clamped to. Default: ``-1.0``.
    """
    
    def __init__(self, min: float = -1.0,  max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, min=self.min, max=self.max)


Clip = Clamp

# endregion


# region Utils

def to_act_layer(act_layer: Any = ReLU(), *args, **kwargs) -> nn.Module:
    """Create activation layer."""
    # if isinstance(norm, str):
    #     norm = LAYER.build(name=norm)
    act_layer = act_layer
    if act_layer is None or not act_layer:
        act_layer = nn.Identity()
    elif isinstance(act_layer, Callable | types.FunctionType | functools.partial):
        act_layer = act_layer(*args, **kwargs)
    return act_layer

# endregion
