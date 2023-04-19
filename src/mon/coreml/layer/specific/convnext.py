#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for ConvNeXt
models.
"""

from __future__ import annotations

__all__ = [
    "ConvNeXtBlock", "ConvNeXtLayer",
]

import functools
from typing import Callable

import torch
import torchvision.ops
from torch import nn

from mon.coreml.layer import base, common
from mon.globals import LAYERS


@LAYERS.register()
class ConvNeXtLayer(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        stage_block_id       : int | None = None,
        total_stage_blocks   : int | None = None,
        norm                 : Callable   = None,
    ):
        super().__init__()
        sd_prob = stochastic_depth_prob
        if (stage_block_id is not None) and (total_stage_blocks is not None):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
        if norm is None:
            norm = functools.partial(common.LayerNorm, eps=1e-6)
        self.block = torch.nn.Sequential(
            common.Conv2d(
                in_channels  = dim,
                out_channels = dim,
                kernel_size  = 7,
                padding      = 3,
                groups       = dim,
                bias         = True,
            ),
            common.Permute([0, 2, 3, 1]),
            norm(dim),
            common.Linear(
                in_features  = dim,
                out_features = 4 * dim,
                bias         = True,
            ),
            common.GELU(),
            common.Linear(
                in_features  = 4 * dim,
                out_features = dim,
                bias         = True,
            ),
            common.Permute([0, 3, 1, 2]),
        )
        self.layer_scale      = torch.nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = torchvision.ops.StochasticDepth(sd_prob, "row")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y  = self.layer_scale * self.block(x)
        y  = self.stochastic_depth(y)
        y += x
        return y


@LAYERS.register()
class ConvNeXtBlock(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        num_layers           : int,
        stage_block_id       : int,
        total_stage_blocks   : int | None = None,
        norm                 : Callable   = None,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                ConvNeXtLayer(
                    dim                   = dim,
                    layer_scale           = layer_scale,
                    stochastic_depth_prob = stochastic_depth_prob,
                    stage_block_id        = stage_block_id,
                    total_stage_blocks    = total_stage_blocks,
                    norm                  = norm,
                )
            )
            stage_block_id += 1
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)
