#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization Layers.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from one.core import Callable
from one.core import NORM_LAYERS

__all__ = [
    "EvoNormBatch2d",
    "EvoNormSample2d",
    "FractionInstanceNorm2d",
    "HalfGroupNorm",
    "HalfInstanceNorm2d",
    "HalfLayerNorm",
    "LayerNorm2d",
    "EvoNormBatch",
    "EvoNormSample",
    "HalfInstanceNorm",
    "FractionInstanceNorm",
]


# MARK: - Modules

@NORM_LAYERS.register(name="evo_norm_batch2d")
class EvoNormBatch2d(nn.Module):

    # MARK: Magic Functions

    def __init__(
        self,
        num_features: int,
        apply_act   : bool               = True,
        momentum    : float              = 0.1,
        eps         : float              = 1e-5,
        drop_block  : Optional[Callable] = None
    ):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.momentum  = momentum
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape),  requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input.")
        x_type = x.dtype
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n   = x.numel() / x.shape[1]
            self.running_var.copy_(
                var.detach() * self.momentum * (n / (n - 1)) +
                self.running_var * (1 - self.momentum)
            )
        else:
            var = self.running_var

        if self.apply_act:
            v     = self.v.to(dtype=x_type)
            x1    = (x.var(dim=(2, 3), unbiased=False, keepdim=True) + self.eps)
            x1    = x1.sqrt().to(dtype=x_type)
            d     = x * v + x1
            d     = d.max((var + self.eps).sqrt().to(dtype=x_type))
            x = x / d
        return x * self.weight + self.bias


@NORM_LAYERS.register(name="evo_norm_sample2d")
class EvoNormSample2d(nn.Module):

    # MARK: Magic Functions

    def __init__(
        self,
        num_features: int,
        apply_act   : bool               = True,
        groups      : int                = 8,
        eps         : float              = 1e-5,
        drop_block  : Optional[Callable] = None
    ):
        super().__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight = nn.Parameter(torch.ones(param_shape),  requires_grad=True)
        self.bias   = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        if apply_act:
            self.v = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.apply_act:
            nn.init.ones_(self.v)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError("Expected 4D input")
        
        b, c, h, w = x.shape
        if c % self.groups != 0:
            raise ValueError()
        
        if self.apply_act:
            n  = x * (x * self.v).sigmoid()
            x  = x.reshape(b, self.groups, -1)
            x1 = (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            x  = n.reshape(b, self.groups, -1) / x1
            x  = x.reshape(b, c, h, w)
        return x * self.weight + self.bias


@NORM_LAYERS.register(name="fraction_instance_norm2d")
class FractionInstanceNorm2d(nn.InstanceNorm2d):
    """Perform fractional measurement normalization.
    
    Args:
        num_features (int):
            Number of input features.
        alpha (float):
            Ratio of input features that will be normalized. Default: `0.5`.
        selection (str):
            Feature selection mechanism. One of: ["linear", "random",
            "interleave"]. Default: `linear`.
            - "linear"    : normalized only first half.
            - "random"    : randomly choose features to normalize.
            - "interleave": interleavingly choose features to normalize.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_features: int,
        alpha       : float = 0.5,
        selection   : str   = "linear",
        affine      : bool  = True,
        *args, **kwargs
    ):
        self.in_channels = num_features
        self.alpha       =  alpha
        self.selection   = selection
        super().__init__(
            num_features=math.ceil(num_features * self.alpha), affine=affine,
            *args, **kwargs
        )

        if self.selection not in ["linear", "random", "interleave"]:
            raise ValueError()
 
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        _, c, _, _ = x.shape
        
        if self.alpha == 0.0:
            return x
        elif self.alpha == 1.0:
            return F.instance_norm(
                x, self.running_mean, self.running_var, self.weight,
                self.bias, self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
        else:
            if self.selection == "random":
                out1_idxes = random.sample(range(self.in_channels), self.num_features)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(x.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(x.device)
                out1       = torch.index_select(x, 1, out1_idxes)
                out2       = torch.index_select(x, 1, out2_idxes)
            elif self.selection == "interleave":
                skip       = int(math.floor(self.in_channels / self.num_features))
                out1_idxes = []
                for i in range(0, self.in_channels, skip):
                    if len(out1_idxes) < self.num_features:
                        out1_idxes.append(i)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                # print(len(out1_idxes), len(out2_idxes), self.num_features)
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(x.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(x.device)
                out1       = torch.index_select(x, 1, out1_idxes)
                out2       = torch.index_select(x, 1, out2_idxes)
            else:  # Half-Half
                split_size = [self.num_features, c - self.num_features]
                out1, out2 = torch.split(x, split_size, dim=1)
            
            out1 = F.instance_norm(
                out1, self.running_mean, self.running_var, self.weight,
                self.bias, self.training or not self.track_running_stats,
                self.momentum, self.eps
            )
            return torch.cat([out1, out2], dim=1)


@NORM_LAYERS.register(name="half_group_norm")
class HalfGroupNorm(nn.GroupNorm):

    # MARK: Magic Functions

    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float = 1e-5,
        affine      : bool  = True,
        *args, **kwargs
    ):
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps,
            affine=affine, *args, **kwargs
        )

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(x, 2, dim=1)
        out_1        = F.group_norm(
            out_1, self.num_groups, self.weight, self.bias, self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


@NORM_LAYERS.register(name="half_instance_norm2d")
class HalfInstanceNorm2d(nn.InstanceNorm2d):
    
    # MARK: Magic Functions
    
    def __init__(self, num_features: int, affine: bool = True, *args, **kwargs):
        super().__init__(
            num_features=num_features // 2, affine=affine, *args, **kwargs
        )
        
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        out_1, out_2 = torch.chunk(x, 2, dim=1)
        out_1        = F.instance_norm(
            out_1, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum,
            self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


@NORM_LAYERS.register(name="half_layer_norm")
class HalfLayerNorm(nn.LayerNorm):

    # MARK: Magic Functions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(x, 2, dim=1)
        out_1        = F.layer_norm(
            out_1, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


@NORM_LAYERS.register(name="layer_norm2d")
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of `2D` spatial [B, C, H, W] tensors."""

    # MARK: Magic Functions

    def __init__(self, num_channels: int, *args, **kwargs):
        super().__init__(num_channels, *args, **kwargs)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
			self.bias, self.eps
        ).permute(0, 3, 1, 2)


# MARK: - Alias

EvoNormBatch         = EvoNormBatch2d
EvoNormSample        = EvoNormSample2d
FractionInstanceNorm = FractionInstanceNorm2d
HalfInstanceNorm     = HalfInstanceNorm2d


# MARK: - Register

NORM_LAYERS.register(name="batch_norm",             module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm1d",           module=nn.BatchNorm1d)
NORM_LAYERS.register(name="batch_norm2d",           module=nn.BatchNorm2d)
NORM_LAYERS.register(name="batch_norm3d",           module=nn.BatchNorm3d)
NORM_LAYERS.register(name="evo_norm_batch",         module=EvoNormBatch)
NORM_LAYERS.register(name="evo_norm_sample",        module=EvoNormSample)
NORM_LAYERS.register(name="fraction_instance_norm", module=HalfInstanceNorm)
NORM_LAYERS.register(name="group_norm",             module=nn.GroupNorm)
NORM_LAYERS.register(name="half_instance_norm",     module=HalfInstanceNorm)
NORM_LAYERS.register(name="layer_norm",             module=nn.LayerNorm)
NORM_LAYERS.register(name="instance_norm",          module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm1d",        module=nn.InstanceNorm1d)
NORM_LAYERS.register(name="instance_norm2d",        module=nn.InstanceNorm2d)
NORM_LAYERS.register(name="instance_norm3d",        module=nn.InstanceNorm3d)
NORM_LAYERS.register(name="sync_batch_norm",        module=nn.SyncBatchNorm)
