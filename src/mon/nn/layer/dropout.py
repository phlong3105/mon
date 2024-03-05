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
from torch.nn.modules.dropout import *
from torchvision.ops import (drop_block2d, drop_block3d, DropBlock2d,
                             DropBlock3d)


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
    

class DropPath(nn.Module):
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
