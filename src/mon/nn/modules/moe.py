#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixture of Experts (MoE) Network.

This module implements the Mixture of Experts (MoE) network.
"""

from __future__ import annotations

__all__ = [
    "LayeredFeatureAggregation",
]

from typing import Sequence

import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from mon import core


# region Layer

class LayeredFeatureAggregation(nn.Module):
    """Layered Feature Aggregation (LFA) Layer fuses features from different
    decoder layers to more robustly and accurately generate the final prediction
    result.
    """
    
    def __init__(
        self,
        in_channels : list[int],
        out_channels: int,
        size        : _size_2_t = None,
    ):
        super().__init__()
        self.in_channels  = core.to_int_list(in_channels)
        self.out_channels = out_channels
        self.num_experts  = len(self.in_channels)
        # Resize
        if size is not None:
            self.size   = core.get_image_size(size)
            self.resize = nn.Upsample(size=self.size, mode="bilinear", align_corners=False)
        else:
            self.size   = None
            self.resize = None
        # Linears
        linears = []
        for in_c in self.in_channels:
            linears.append(nn.Conv2d(in_c, self.out_channels, 1))
        self.linears = nn.ModuleList(linears)
        # Conv & softmax
        self.conv    = nn.Conv2d(self.out_channels * self.num_experts, self.out_channels, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(input) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} input tensors, "
                             f"but got {len(input)}")
        r = []
        for i, inp in enumerate(input):
            if self.resize is not None:
                r.append(self.linears[i](self.resize(inp)))
            else:
                r.append(self.linears[i](inp))
        o_s = torch.cat(r, dim=1)
        w   = self.softmax(self.conv(o_s))
        o_w = torch.stack([r[i] * w[:, i] for i, _ in enumerate(r)], dim=1)
        o   = torch.sum(o_w, dim=1)
        return o

# endregion
