#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements drop-out layers."""

from __future__ import annotations

__all__ = [
    "AlphaDropout",
    "DropBlock2d",
    "DropBlock3d",
    "DropPath",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "FeatureAlphaDropout",
    "drop_block2d",
    "drop_block3d",
]

import torch
from torch import nn
from torchvision import ops

from mon.globals import LAYERS
from mon.nn.layer import base


# region Drop Block

@LAYERS.register()
class DropBlock2d(base.PassThroughLayerParsingMixin, ops.DropBlock2d):
    pass


@LAYERS.register()
class DropBlock3d(base.PassThroughLayerParsingMixin, ops.DropBlock3d):
    pass


drop_block2d = ops.drop_block2d
drop_block3d = ops.drop_block3d


# endregion


# region Drop Out

@LAYERS.register()
class AlphaDropout(base.PassThroughLayerParsingMixin, nn.AlphaDropout):
    pass


@LAYERS.register()
class Dropout(base.PassThroughLayerParsingMixin, nn.Dropout):
    pass


@LAYERS.register()
class Dropout1d(base.PassThroughLayerParsingMixin, nn.Dropout1d):
    pass


@LAYERS.register()
class Dropout2d(base.PassThroughLayerParsingMixin, nn.Dropout2d):
    pass


@LAYERS.register()
class Dropout3d(base.PassThroughLayerParsingMixin, nn.Dropout3d):
    pass


@LAYERS.register()
class FeatureAlphaDropout(base.PassThroughLayerParsingMixin, nn.FeatureAlphaDropout):
    pass


# endregion


# region Drop Path

def drop_path(
    input        : torch.Tensor,
    p            : float = 0.0,
    training     : bool  = False,
    scale_by_keep: bool  = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    
    References:
        `<https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py>`__

    Args:
        input: Input.
        p: Probability of the path to be zeroed. Default: ``0.0``.
        training: Is in training run?. Default: ``False``.
    """
    x = input
    if p == 0.0 or not training:
        return x
    keep_prob     = 1 - p
    shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
    

@LAYERS.register()
class DropPath(base.PassThroughLayerParsingMixin, nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Args:
        p: Probability of the path to be zeroed. Default: ``0.1``.
    """
    
    def __init__(self, p: float = 0.1, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob     = p
        self.scale_by_keep = scale_by_keep
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = drop_path(
            input         = x,
            p             = self.drop_prob,
            training      = self.training,
            scale_by_keep = self.scale_by_keep,
        )
        return y

# endregion
