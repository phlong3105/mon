#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements normalization layers."""

from __future__ import annotations

__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm2dAct",
    "BatchNorm2dReLU",
    "BatchNorm3d",
    "CrossMapLRN2d",
    "FractionalInstanceNorm2d",
    "FractionalInstanceNorm2dOld",
    "FrozenBatchNorm2d",
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
    "LocalResponseNorm",
    "SyncBatchNorm",
]

import random
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import misc

from mon.core import math
from mon.globals import LAYERS
from mon.nn.layer import activation, base, linear


# region Batch Normalization

@LAYERS.register()
class BatchNorm2dAct(base.SameChannelsLayerParsingMixin, nn.BatchNorm2d):
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
        act                : Callable = activation.ReLU(),
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
        self.act = activation.to_act_layer(act, inplace)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = super().forward(x)
        if self.act is not None:
            y = self.act(y)
        return y


@LAYERS.register()
class BatchNorm2dReLU(BatchNorm2dAct):
    """BatchNorm2d + ReLU.

    This module performs BatchNorm2d + ReLU in a manner that will remain
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
            act                 = activation.ReLU(),
            inplace             = inplace,
            drop_block          = drop_block
        )


@LAYERS.register()
class BatchNorm1d(base.SameChannelsLayerParsingMixin, nn.BatchNorm1d):
    pass


@LAYERS.register()
class BatchNorm2d(base.SameChannelsLayerParsingMixin, nn.BatchNorm2d):
    pass


@LAYERS.register()
class BatchNorm3d(base.SameChannelsLayerParsingMixin, nn.BatchNorm3d):
    pass


@LAYERS.register()
class LazyBatchNorm1d(base.SameChannelsLayerParsingMixin, nn.LazyBatchNorm1d):
    pass


@LAYERS.register()
class LazyBatchNorm2d(base.SameChannelsLayerParsingMixin, nn.LazyBatchNorm2d):
    pass


@LAYERS.register()
class LazyBatchNorm3d(base.SameChannelsLayerParsingMixin, nn.LazyBatchNorm3d):
    pass


@LAYERS.register()
class SyncBatchNorm(base.SameChannelsLayerParsingMixin, nn.SyncBatchNorm):
    pass


# endregion


# region Instance Normalization

@LAYERS.register()
class FractionalInstanceNorm2dOld(base.SameChannelsLayerParsingMixin, nn.InstanceNorm2d):
    """Fractional Instance Normalization is a generalization of Half Instance
    Normalization.
    
    Args:
        num_features: Number of input features.
        ratio: Ratio of input features that will be normalized. Default:
            ``0.5``.
        selection: Feature selection mechanism. One of: ``'linear'``,
            ``'random'``, or ``'interleave'``
                - ``'linear'``    : normalized only first half.
                - ``'random'``    : randomly choose features to normalize.
                - ``'interleave'``: interleaving choose features to normalize.
            Default: ``linear``.
    """
    
    def __init__(
        self,
        num_features       : int,
        ratio              : float = 0.5,
        selection          : str   = "linear",
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
    ):
        self.in_channels = num_features
        self.ratio       = ratio
        self.selection   = selection
        super().__init__(
            num_features        = math.ceil(num_features * self.ratio),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )
        
        if self.selection not in ["linear", "random", "interleave"]:
            raise ValueError(f"{self.selection}")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        self._check_input_dim(x)
        _, c, _, _ = x.shape
        
        if self.ratio == 0.0:
            return x
        elif self.ratio == 1.0:
            y = functional.instance_norm(
                input           = x,
                running_mean    = self.running_mean,
                running_var     = self.running_var,
                weight          = self.weight,
                bias            = self.bias,
                use_input_stats = self.training or not self.track_running_stats,
                momentum        = self.momentum,
                eps             = self.eps
            )
            return y
        else:
            if self.selection == "random":
                y1_idxes = random.sample(range(self.in_channels), self.num_features)
                y2_idxes = list(set(range(self.in_channels)) - set(y1_idxes))
                y1_idxes = torch.Tensor(y1_idxes).to(torch.int).to(x.device)
                y2_idxes = torch.Tensor(y2_idxes).to(torch.int).to(x.device)
                y1       = torch.index_select(x, 1, y1_idxes)
                y2       = torch.index_select(x, 1, y2_idxes)
            elif self.selection == "interleave":
                skip     = int(math.floor(self.in_channels / self.num_features))
                y1_idxes = []
                for i in range(0, self.in_channels, skip):
                    if len(y1_idxes) < self.num_features:
                        y1_idxes.append(i)
                y2_idxes = list(set(range(self.in_channels)) - set(y1_idxes))
                y1_idxes = torch.Tensor(y1_idxes).to(torch.int).to(x.device)
                y2_idxes = torch.Tensor(y2_idxes).to(torch.int).to(x.device)
                y1       = torch.index_select(x, 1, y1_idxes)
                y2       = torch.index_select(x, 1, y2_idxes)
            else:  # Half-Half
                split_size = [self.num_features, c - self.num_features]
                y1, y2     = torch.split(x, split_size, dim=1)
            
            y1 = functional.instance_norm(
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


@LAYERS.register()
class FractionalInstanceNorm2d(base.SameChannelsLayerParsingMixin, nn.InstanceNorm2d):
    """Apply Instance Normalization on a fraction of the input tensor.
    
    Args:
        num_features: Number of input features.
        p: Ratio of input features that will be normalized. Default: ``0.5``.
        scheme: Feature selection mechanism. One of:
            - ``'half'``        : Split the input tensor into two even parts.
                                  Normalized the first half.
            - ``'bipartite'``   : Split the input tensor into two uneven parts.
                                  Normalized the first half.
            - ``'checkerboard'``: Normalized the input tensor following the
                                  checkerboard pattern.
            - ``'random'``      : Normalized the input tensor in randomly.
            - ``'adaptive'``    : Define a learnable weight parameter. Then
                                  apply weighted sum between the normalized
                                  tensor and the original tensor.
            - ``'attentive'``   : Apply channel attention to determine the
                                  channels' weights. Then apply weighted sum
                                  between the normalized tensor and the original
                                  tensor.
            Default to ``'half'``.
        pool: Pooling type. One of: ``'avg'``, or ``'max'``. Default: ``'avg'``.
        bias: Add bias for ``adaptive`` scheme. Default: ``True``.
    """
    
    schemes = [
        "half", "bipartite", "checkerboard", "random", "adaptive",
        "attentive",
    ]
    
    def __init__(
        self,
        num_features       : int,
        p                  : float          = 0.5,
        scheme             : str            = "half",
        pool               : str | Callable = "avg",
        bias               : bool           = True,
        eps                : float          = 1e-5,
        momentum           : float          = 0.1,
        affine             : bool           = True,
        track_running_stats: bool           = False,
        device             : Any            = None,
        dtype              : Any            = None,
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
        if scheme not in self.schemes:
            raise ValueError(f"``scheme`` must be one of: {self.schemes}. But got: {scheme}.")
        if scheme == "half":
            self.alpha = torch.zeros(num_features)
            self.alpha[0:math.ceil(num_features * 0.5)] = 1
        elif scheme == "bipartite":
            self.alpha = torch.zeros(num_features)
            self.alpha[0:math.ceil(num_features * p)] = 1
        elif scheme == "checkerboard":
            in_channels = math.ceil(num_features * p)
            step_size  = int(math.floor(in_channels / num_features))
            self.alpha = torch.zeros(num_features)
            for i in range(0, in_channels, step_size):
                self.alpha[i] = 1
        elif scheme == "random":
            in_channels = math.ceil(num_features * p)
            rand        = random.sample(range(in_channels), num_features)
            self.alpha  = torch.zeros(num_features)
            for i in rand:
                self.alpha[i] = 1
        elif scheme == "adaptive":
            self.alpha = torch.nn.Parameter(torch.full([num_features], p))
        elif scheme == "attentive":
            if pool not in ["avg", "max"]:
                raise ValueError(f"``pool`` must be one of: [``'avg'``, ``'max'``], but got {pool}.")
            self.channel_attention = nn.Sequential(
                self.Flatten(),
                linear.Linear(
                    in_features  = num_features,
                    out_features = math.ceil(num_features * p),
                ),
                activation.ReLU(),
                linear.Linear(
                    in_features  = math.ceil(num_features * p),
                    out_features = num_features,
                )
            )
        if bias:
            self.beta1 = torch.nn.Parameter(torch.zeros(num_features))
            self.beta2 = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.beta1 = None
            self.beta2 = None
        
        self.p      = p
        self.scheme = scheme
        self.pool   = pool
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        x          = input
        b, c, h, w = x.shape
        x_norm     = functional.instance_norm(
            input           = x,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        
        if self.scheme in ["half", "bipartite", "checkerboard", "random"]:
            alpha = self.alpha.reshape(-1, c, 1, 1).to(x.device)
            y     = (x_norm * alpha) + (x * (1 - alpha))
        elif self.scheme in ["adaptive"]:
            alpha = self.alpha.reshape(-1, c, 1, 1).to(x.device)
            if self.beta1 is not None and self.beta2 is not None:
                beta1 = self.beta1.reshape(-1, c, 1, 1).to(x.device)
                beta2 = self.beta2.reshape(-1, c, 1, 1).to(x.device)
                y     = (x_norm * alpha + beta1) + (x * (1 - alpha) + beta2)
            else:
                y = (x_norm * alpha) + (x * (1 - alpha))
        elif self.scheme in ["attentive"]:
            if self.pool == "avg":
                pool = functional.avg_pool2d(
                    input       = x,
                    kernel_size = (x.size(2), x.size(3)),
                    stride      = (x.size(2), x.size(3)),
                )
            else:
                pool = functional.max_pool2d(
                    input       = x,
                    kernel_size = (x.size(2), x.size(3)),
                    stride      = (x.size(2), x.size(3)),
                )
            alpha = self.channel_attention(pool)
            alpha = torch.sigmoid(alpha).unsqueeze(2).unsqueeze(3).expand_as(x)
            if self.beta1 is not None and self.beta2 is not None:
                beta1 = self.beta1.reshape(-1, c, 1, 1).to(x.device)
                beta2 = self.beta2.reshape(-1, c, 1, 1).to(x.device)
                y = (x_norm * alpha + beta1) + (x * (1 - alpha) + beta2)
            else:
                y = (x_norm * alpha) + (x * (1 - alpha))
        else:
            y = x_norm
        return y


@LAYERS.register()
class HalfInstanceNorm2d(base.SameChannelsLayerParsingMixin, nn.InstanceNorm2d):
    
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
        y1 = functional.instance_norm(
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
 

@LAYERS.register()
class InstanceNorm1d(base.SameChannelsLayerParsingMixin, nn.InstanceNorm1d):
    pass


@LAYERS.register()
class InstanceNorm2d(base.SameChannelsLayerParsingMixin, nn.InstanceNorm2d):
    pass


@LAYERS.register()
class InstanceNorm3d(base.SameChannelsLayerParsingMixin, nn.InstanceNorm3d):
    pass


@LAYERS.register()
class LazyInstanceNorm1d(base.SameChannelsLayerParsingMixin, nn.LazyInstanceNorm1d):
    pass


@LAYERS.register()
class LazyInstanceNorm2d(base.SameChannelsLayerParsingMixin, nn.LazyInstanceNorm2d):
    pass


@LAYERS.register()
class LazyInstanceNorm3d(base.SameChannelsLayerParsingMixin, nn.LazyInstanceNorm3d):
    pass


# endregion


# region Group Normalization

@LAYERS.register()
class GroupNorm(base.SameChannelsLayerParsingMixin, nn.GroupNorm):
    pass


@LAYERS.register()
class GroupNormAct(GroupNorm):
    """GroupNorm + Activation.

    This module performs GroupNorm + Activation in a manner that will remain
    backwards compatible with weights trained with separate gn, norm. This is why
    we inherit from GN instead of composing it as a .gn member.
    """
    
    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float    = 1e-5,
        affine      : bool     = True,
        device      : Any      = None,
        dtype       : Any      = None,
        act         : Callable = activation.ReLU,
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
        self.act = activation.to_act_layer(act, inplace)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = functional.group_norm(
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

@LAYERS.register()
class LayerNorm(base.SameChannelsLayerParsingMixin, nn.LayerNorm):
    pass


@LAYERS.register()
class LayerNorm2d(base.SameChannelsLayerParsingMixin, nn.LayerNorm):
    """LayerNorm for channels of 2D spatial [B, C, H, W] tensors."""
    
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
        y = functional.layer_norm(
            input            = x.permute(0, 2, 3, 1),
            normalized_shape = self.normalized_shape,
            weight           = self.weight,
            bias             = self.bias,
            eps              = self.eps
        ).permute(0, 3, 1, 2)
        return y

# endregion


# region Misc

@LAYERS.register()
class CrossMapLRN2d(base.SameChannelsLayerParsingMixin, nn.CrossMapLRN2d):
    pass


@LAYERS.register()
class FrozenBatchNorm2d(base.SameChannelsLayerParsingMixin, misc.FrozenBatchNorm2d):
    pass


@LAYERS.register()
class LocalResponseNorm(base.SameChannelsLayerParsingMixin, nn.LocalResponseNorm):
    pass

# endregion
