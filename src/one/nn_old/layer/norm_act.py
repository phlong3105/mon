#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Normalization + Activation Layers.
"""

from __future__ import annotations

import functools
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from one.core import Callable
from one.core import NORM_ACT_LAYERS
from one.nn.layer.act import create_act_layer
from one.nn.layer.inplace_abn import InplaceAbn
from one.nn.layer.norm import EvoNormBatch2d
from one.nn.layer.norm import EvoNormSample2d

__all__ = [
    "convert_norm_act",
    "create_norm_act",
    "get_norm_act_layer",
    "BatchNormAct2d",
    "BatchNormReLU2d",
    "GroupNormAct",
    "BatchNormReLU",
    "BatchNormAct",
]


# MARK: - Modules

@NORM_ACT_LAYERS.register(name="bn_act2d")
@NORM_ACT_LAYERS.register(name="batch_norm_act2d")
class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation.

    This module performs BatchNorm + Activation in a manner that will remain
    backwards compatible with weights trained with separate bn, act. This is
    why we inherit from BN instead of composing it as a `.bn` member.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_features       : int,
        eps                : float              = 1e-5,
        momentum           : float              = 0.1,
        affine             : bool               = True,
        track_running_stats: bool               = True,
        apply_act          : bool               = True,
        act_layer          : Optional[Callable] = nn.ReLU,
        inplace            : bool               = True,
        drop_block         : Optional[Callable] = None
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.act = create_act_layer(apply_act, act_layer, inplace)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        # FIXME cannot call parent forward() and maintain jit.script
        # compatibility?
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x

    def _forward_jit(self, x: Tensor) -> Tensor:
        """A cut & paste of the contents of the PyTorch BatchNorm2d forward
        function.
        """
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting
            #  this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = \
                        1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        x = F.batch_norm(
            x, self.running_mean, self.running_var, self.weight,  self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps
        )
        return x

    @torch.jit.ignore
    def _forward_python(self, x: Tensor) -> Tensor:
        return super(BatchNormAct2d, self).forward(x)


@NORM_ACT_LAYERS.register(name="bn_relu2d")
@NORM_ACT_LAYERS.register(name="batch_norm_relu2d")
class BatchNormReLU2d(BatchNormAct2d):
    """BatchNorm + ReLU.

    This module performs BatchNorm + ReLU in a manner that will remain
    backwards compatible with weights trained with separate bn, act. This is
    why we inherit from BN instead of composing it as a .bn member.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_features       : int,
        eps                : float              = 1e-5,
        momentum           : float              = 0.1,
        affine             : bool               = True,
        track_running_stats: bool               = True,
        apply_act          : bool               = True,
        inplace            : bool               = True,
        drop_block         : Optional[Callable] = None
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            apply_act           = apply_act,
            act_layer           = nn.ReLU,
            inplace             = inplace,
            drop_block          = drop_block
        )


@NORM_ACT_LAYERS.register(name="gn_act")
@NORM_ACT_LAYERS.register(name="group_norm_act")
class GroupNormAct(nn.GroupNorm):
    """GroupNorm + Activation.

    This module performs GroupNorm + Activation in a manner that will remain
    backwards compatible with weights trained with separate gn, act. This is
    why we inherit from GN instead of composing it as a .gn member.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_channels: int,
        num_groups  : int,
        eps         : float              = 1e-5,
        affine      : bool               = True,
        apply_act   : bool               = True,
        act_layer   : Optional[Callable] = nn.ReLU,
        inplace     : bool               = True,
        drop_block  : Optional[Callable] = None
    ):
        # NOTE num_channel and num_groups order flipped for easier layer
        # swaps / binding of fixed args
        super().__init__(num_groups, num_channels, eps, affine)
        self.act = create_act_layer(apply_act, act_layer, inplace)
    
    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.act(x)
        return x


# MARK: - Alias

BatchNormAct  = BatchNormAct2d
BatchNormReLU = BatchNormReLU2d


# MARK: - Register

NORM_ACT_LAYERS.register(name="bn_act",          module=BatchNormAct)
NORM_ACT_LAYERS.register(name="batch_norm_act",  module=BatchNormAct)
NORM_ACT_LAYERS.register(name="bn_relu",         module=BatchNormReLU)
NORM_ACT_LAYERS.register(name="batch_norm_relu", module=BatchNormReLU)


# MARK: - Builder

_NORM_ACT_TYPES        = {BatchNormAct2d, GroupNormAct, EvoNormBatch2d,
                          EvoNormSample2d, InplaceAbn}
# Requires act_layer arg to define act type
_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d, GroupNormAct, InplaceAbn}


def create_norm_act(
    layer_type  : str,
    num_features: int,
    apply_act   : bool = True,
    jit         : bool = False,
    **kwargs
) -> nn.Module:
    layer_parts = layer_type.split("_")  # e.g. batchnorm-leaky_relu
    if len(layer_parts) not in (1, 2):
        raise ValueError
    layer = get_norm_act_layer(layer_parts[0])
    #activation_class = layer_parts[1].lower() if len(layer_parts) > 1 else ''
    # FIXME support string act selection?
    layer_instance = layer(num_features, apply_act=apply_act, **kwargs)
    if jit:
        layer_instance = torch.jit.script(layer_instance)
    return layer_instance


def get_norm_act_layer(layer_class: str):
    layer_class = layer_class.replace("_", "").lower()
    if layer_class.startswith("batchnorm"):
        layer = BatchNormAct2d
    elif layer_class.startswith("groupnorm"):
        layer = GroupNormAct
    elif layer_class == "evonormbatch":
        layer = EvoNormBatch2d
    elif layer_class == "evonormsample":
        layer = EvoNormSample2d
    elif layer_class == "iabn" or layer_class == "inplaceabn":
        layer = InplaceAbn
    elif True:
        raise ValueError("Invalid norm_act layer (%s)" % layer_class)
    return layer


def convert_norm_act(norm_layer: Callable, act_layer: Callable) -> nn.Module:
    if not isinstance(norm_layer, Callable):
        raise ValueError
    if not (act_layer is None or isinstance(act_layer, Callable)):
        raise ValueError
    norm_act_kwargs = {}

    # Unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, functools.partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        norm_act_layer = NORM_ACT_LAYERS.build(name=norm_layer)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith("batchnorm"):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith("groupnorm"):
            norm_act_layer = GroupNormAct
        elif True:
            raise ValueError(f"No equivalent norm_act layer for {type_name}.")

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None`
        # implies no activation. In the future, may force use of `apply_act`
        # with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault("act_layer", act_layer)
    if norm_act_kwargs:
        # bind/rebind args
        norm_act_layer = functools.partial(norm_act_layer, **norm_act_kwargs)
    return norm_act_layer
