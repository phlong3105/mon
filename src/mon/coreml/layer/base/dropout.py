#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements drop-out layers."""

from __future__ import annotations

__all__ = [
    "AlphaDropout", "drop_block2d", "drop_block3d", "DropBlock2d",
    "DropBlock3d", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "DropPath",
    "FeatureAlphaDropout",
]

import torch
from torch import nn
from torchvision import ops

from mon.coreml.layer.base import base
from mon.globals import LAYERS


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
    input   : torch.Tensor,
    p       : float = 0.0,
    training: bool  = False,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks). We follow the implementation:
    https://github.com/rwightman/pytorch-image-models/blob
    /a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    Args:
        input: Input.
        p: Probability of the path to be zeroed. Defaults to 0.0.
        training: Is in training run?. Defaults to False.
    """
    x = input
    if p == 0.0 or not training:
        return x
    keep_prob     = 1 - p
    shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device))
    y             = x.div(keep_prob) * random_tensor.floor()
    return y


@LAYERS.register()
class DropPath(base.PassThroughLayerParsingMixin, nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Args:
        p: Probability of the path to be zeroed. Defaults to 0.1.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.drop_prob = p
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = drop_path(
            input    = x,
            p        = self.drop_prob,
            training = self.training,
        )
        return y

# endregion
