#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization Layers.

This module implements normalization layers.
"""

from __future__ import annotations

__all__ = [
    "AdaptiveBatchNorm2d",
    "AdaptiveInstanceNorm2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm2dAct",
    "BatchNorm2dReLU",
    "BatchNorm3d",
    "CrossMapLRN2d",
    "GroupNorm",
    "GroupNormAct",
    "HalfInstanceNorm2d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LayerNorm2d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LearnableInstanceNorm2d",
    "LocalResponseNorm",
    "SyncBatchNorm",
]

import math
import random
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import *
from torch.nn.modules.instancenorm import *
from torch.nn.modules.normalization import *

from mon.nn.modules import activation, linear


# region Batch Normalization

class AdaptiveBatchNorm2d(nn.Module):
    """Adaptive Batch Normalization.
    
    References:
        - https://arxiv.org/abs/1709.00643
        - https://github.com/nrupatunga/Fast-Image-Filters
    """
    
    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
    ):
        super().__init__()
        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))
        self.bn  = nn.BatchNorm2d(num_features, eps, momentum)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.w_0 * input + self.w_1 * self.bn(input)
    

class BatchNorm2dAct(nn.BatchNorm2d):
    """BatchNorm2d + Activation.
    
    This module performs BatchNorm2d + Activation in a manner that will remain
    backwards compatible with weights trained with separate bn, norm. This is why
    we inherit from BN instead of composing it as a .bn member.
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float    = 1e-5,
        momentum           : float    = 0.1,
        affine             : bool     = True,
        track_running_stats: bool     = True,
        device             : Any      = None,
        dtype              : Any      = None,
        act_layer          : Callable = activation.ReLU(),
        inplace            : bool     = True,
        *args, **kwargs
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype
        )
        self.act = activation.to_act_layer(act_layer, inplace)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = super().forward(x)
        if self.act:
            y = self.act(y)
        return y


class BatchNorm2dReLU(BatchNorm2dAct):
    """BatchNorm2d + ReLU.

    This module performs BatchNorm2d + ReLU in a manner that will remain
    backwards compatible with weights trained with separate bn, norm. This is
    why we inherit from BN instead of composing it as a .bn member.
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float    = 1e-5,
        momentum           : float    = 0.1,
        affine             : bool     = True,
        track_running_stats: bool     = True,
        device             : Any      = None,
        dtype              : Any      = None,
        inplace            : bool     = True,
        drop_block         : Callable = None,
        *args, **kwargs
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
            act_layer           = activation.ReLU(),
            inplace             = inplace,
            drop_block          = drop_block
        )

# endregion


# region Instance Normalization

class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization adopted from :obj:`AdaptiveBatchNorm2d`.
    """
    
    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        affine      : bool  = False,
    ):
        super().__init__()
        self.w_0  = nn.Parameter(torch.Tensor([1.0]))
        self.w_1  = nn.Parameter(torch.Tensor([0.0]))
        self.norm = nn.InstanceNorm2d(num_features, eps, momentum, affine)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.w_0 * input + self.w_1 * self.bn(input)
    

class LearnableInstanceNorm2d(nn.InstanceNorm2d):
    """Apply Instance Normalization on a fraction of the input tensor.
    The number of normalized features is learnable during training.
    
    Args:
        num_features: Number of features of the input tensor.
        r: Fraction of the input tensor to be normalized. Default: ``0.5``.
    """
   
    def __init__(
        self,
        num_features       : int,
        r                  : float = 0.5,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )
        self.r = nn.Parameter(torch.full([num_features], r), requires_grad=True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        x          = input
        b, c, h, w = x.shape
        x_norm     = F.instance_norm(
            input           = x,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        r = self.r.reshape(-1, c, 1, 1)
        y = (x_norm * r) + (x * (1 - r))
        return y


class HalfInstanceNorm2d(nn.InstanceNorm2d):
    """Apply Instance Normalization on the first half of the input tensor.
    
    Args:
        num_features: Number of features of the input tensor.
        eps: Small constant for numerical stability. Default: ``1e-5``.
        momentum: Momentum for moving average. Default: ``0.1``.
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
    ):
        super().__init__(
            num_features        = math.ceil(num_features / 2),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        self._check_input_dim(x)
        if x.dim() == 3:
            y1, y2 = torch.chunk(x, 2, dim=0)
        elif x.dim() == 4:
            y1, y2 = torch.chunk(x, 2, dim=1)
        else:
            raise ValueError
        y1 = F.instance_norm(
            input           = y1,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        return torch.cat([y1, y2], dim=1)
 
# endregion


# region Group Normalization

class GroupNormAct(GroupNorm):
    """GroupNorm + Activation.

    This module performs GroupNorm + Activation in a manner that will remain
    backwards compatible with weights trained with separate gn, norm. This is
    why we inherit from GN instead of composing it as a .gn member.
    """
    
    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float    = 1e-5,
        affine      : bool     = True,
        device      : Any      = None,
        dtype       : Any      = None,
        act_layer   : Callable = activation.ReLU,
        inplace     : bool     = True,
    ):
        super().__init__(
            num_groups   = num_groups,
            num_channels = num_channels,
            eps          = eps,
            affine       = affine,
            device       = device,
            dtype        = dtype
        )
        self.act = activation.to_act_layer(act_layer, inplace)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = F.group_norm(
            input      = x,
            num_groups = self.num_groups,
            weight     = self.weight,
            bias       = self.bias,
            eps        = self.eps
        )
        y = self.act(y)
        return y

# endregion


# region Layer Normalization

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of 2D spatial ``[B, C, H, W]`` tensors."""
    
    def __init__(
        self,
        normalized_shape  : Any,
        eps               : float = 1e-5,
        elementwise_affine: bool  = True,
        device            : Any   = None,
        dtype             : Any   = None,
    ):
        super().__init__(
            normalized_shape   = normalized_shape,
            eps                = eps,
            elementwise_affine = elementwise_affine,
            device             = device,
            dtype              = dtype
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = F.layer_norm(
            input            = x.permute(0, 2, 3, 1),
            normalized_shape = self.normalized_shape,
            weight           = self.weight,
            bias             = self.bias,
            eps              = self.eps
        ).permute(0, 3, 1, 2)
        return y

# endregion
