#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common Activation Layers.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from one.core import ACT_LAYERS
from one.core import Callable
from one.core import Int2T
from one.core import to_2tuple

__all__ = [
    "create_act_layer",
    "gelu",
    "hard_mish",
    "hard_swish_yolov4",
    "mish",
    "sigmoid",
    "swish",
    "tanh",
    "ArgMax", 
    "Clamp", 
    "Clip",
    "FReLU", 
    "GELU",
    "HardMish",
    "HardSwishYoloV4",
    "MemoryEfficientMish",
    "MemoryEfficientSwish",
    "Mish",
    "PReLU",
    "Sigmoid",
    "Swish",
    "Tanh",
]


# MARK: - Functional

def gelu(x: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return F.gelu(x)


def hard_mish(x: Tensor, inplace: bool = False) -> Tensor:
    """Hard Mish Experimental, based on notes by Mish author Diganta Misra at
    https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


def hard_swish_yolov4(x: Tensor, inplace: bool = False) -> Tensor:
    return x * F.hardtanh(x + 3, 0.0, 6.0, inplace) / 6.0


def mish(x: Tensor, inplace: bool = False) -> Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


def sigmoid(x: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return x.sigmoid_() if inplace else x.sigmoid()


def swish(x: Tensor, inplace: bool = False) -> Tensor:
    """Swish described in: https://arxiv.org/abs/1710.05941"""
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


def tanh(x: Tensor, inplace: bool = False) -> Tensor:
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    return x.tanh_() if inplace else x.tanh()


# MARK: - Modules

@ACT_LAYERS.register(name="arg_max")
class ArgMax(nn.Module):
    """Find the indices of the maximum value of all elements in the input
    image.
    
    Attributes:
        dim (int, optional):
            Dimension to find the indices of the maximum value.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.argmax(x, dim=self.dim)


@ACT_LAYERS.register(name="clamp")
class Clamp(nn.Module):
    """Clamp activation layer. This activation function is to clamp the feature
    map value within :math:`[min, max]`. More details can be found in
    `torch.clamp()`.
    
    Attributes:
        min (float):
            Lower-bound of the range to be clamped to.
        max (float):
            Upper-bound of the range to be clamped to.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, min: float = -1.0, max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, min=self.min, max=self.max)


@ACT_LAYERS.register(name="frelu")
class FReLU(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, c1: int, k: Int2T = 3):
        super().__init__()
        k         = to_2tuple(k)
        self.conv = nn.Conv2d(c1, c1, k, (1, 1), 1, groups=c1)
        self.bn   = nn.BatchNorm2d(c1)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, self.bn(self.conv(x)))


@ACT_LAYERS.register(name="gelu", force=True)
class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg).
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x, self.inplace)


@ACT_LAYERS.register(name="hard_mish")
class HardMish(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return hard_mish(x, self.inplace)


@ACT_LAYERS.register(name="hard_swish_yolov4")
class HardSwishYoloV4(nn.Module):

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return hard_swish_yolov4(x, self.inplace)
  

@ACT_LAYERS.register(name="memory_efficient_mish")
class MemoryEfficientMish(nn.Module):
    
    # noinspection PyMethodOverriding
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # input * tanh(ln(1 + exp(input)))

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.saved_tensors[0]
            sx = torch.sigmoid(input)
            fx = F.softplus(input).tanh()
            return grad_output * (fx + input * sx * (1 - fx * fx))

    def forward(self, x: Tensor) -> Tensor:
        return self.F.apply(x)
    
    
@ACT_LAYERS.register(name="memory_efficient_swish")
class MemoryEfficientSwish(nn.Module):
    
    # noinspection PyMethodOverriding
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            input  = ctx.saved_tensors[0]
            sx = torch.sigmoid(input)
            return grad_output * (sx * (1 + input * (1 - sx)))

    def forward(self, x: Tensor) -> Tensor:
        return self.F.apply(x)


@ACT_LAYERS.register(name="mish")
class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function -
    https://arxiv.org/abs/1908.08681
    """

    # MARK: Magic Functions

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return mish(x)


@ACT_LAYERS.register(name="prelu", force=True)
class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_parameters: int   = 1,
        init          : float = 0.25,
        inplace       : bool  = False
    ):
        super().__init__(num_parameters=num_parameters, init=init)
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return F.prelu(x, self.weight)


@ACT_LAYERS.register(name="sigmoid", force=True)
class Sigmoid(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid_() if self.inplace else x.sigmoid()


@ACT_LAYERS.register(name="swish")
class Swish(nn.Module):
    """Swish Module. This module applies the swish function."""
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return swish(x, self.inplace)
    

@ACT_LAYERS.register(name="tanh", force=True)
class Tanh(nn.Module):
    """PyTorch has this, but not with a consistent inplace arguments interface.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh_() if self.inplace else x.tanh()
  
  
# MARK: - Alias

Clip = Clamp


# MARK: - Register

ACT_LAYERS.register(name="celu",         module=nn.CELU)
ACT_LAYERS.register(name="clip",         module=Clip)
ACT_LAYERS.register(name="elu",          module=nn.ELU)
ACT_LAYERS.register(name="gelu",         module=nn.GELU)
ACT_LAYERS.register(name="hard_shrink",  module=nn.Hardshrink)
ACT_LAYERS.register(name="hard_sigmoid", module=nn.Hardsigmoid)
ACT_LAYERS.register(name="hard_swish", 	 module=nn.Hardswish)
ACT_LAYERS.register(name="hard_tanh",    module=nn.Hardtanh)
ACT_LAYERS.register(name="identity",     module=nn.Identity)
ACT_LAYERS.register(name="leaky_relu",   module=nn.LeakyReLU)
ACT_LAYERS.register(name="log_sigmoid",  module=nn.LogSigmoid)
ACT_LAYERS.register(name="log_softmax",  module=nn.LogSoftmax)
ACT_LAYERS.register(name="prelu",        module=nn.PReLU)
ACT_LAYERS.register(name="relu", 		 module=nn.ReLU)
ACT_LAYERS.register(name="relu6", 		 module=nn.ReLU6)
ACT_LAYERS.register(name="rrelu", 		 module=nn.RReLU)
ACT_LAYERS.register(name="selu", 		 module=nn.SELU)
ACT_LAYERS.register(name="sigmoid",		 module=nn.Sigmoid)
ACT_LAYERS.register(name="silu", 		 module=nn.SiLU)
ACT_LAYERS.register(name="softmax",      module=nn.Softmax)
ACT_LAYERS.register(name="softmin",      module=nn.Softmin)
ACT_LAYERS.register(name="softplus", 	 module=nn.Softplus)
ACT_LAYERS.register(name="softshrink",   module=nn.Softshrink)
ACT_LAYERS.register(name="softsign",     module=nn.Softsign)
ACT_LAYERS.register(name="tanh",		 module=nn.Tanh)
ACT_LAYERS.register(name="tanhshrink",   module=nn.Tanhshrink)


# MARK: - Builder

def create_act_layer(
    apply_act: bool               = True,
    act_layer: Optional[Callable] = nn.ReLU(),
    inplace  : bool               = True,
    **_
) -> nn.Module:
    """Create activation layer."""
    if isinstance(act_layer, str):
        act_layer = ACT_LAYERS.build(name=act_layer)
    if isinstance(act_layer, types.FunctionType):
        act_args  = dict(inplace=True) if inplace else {}
        act_layer = act_layer(**act_args)
    if act_layer is None or not apply_act:
        act_layer = nn.Identity()
    return act_layer
