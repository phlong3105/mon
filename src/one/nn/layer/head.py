#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Head Layers.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.nn import functional as F

from one.core import HEADS
from one.nn.layer.linear import Linear
from one.nn.layer.pool import SelectAdaptivePool2d

__all__ = [
    "create_classifier",
    "Classifier",
    "Detector",
]


# MARK: - Functional

def _create_pool(
    num_features: int,
    num_classes : int,
    pool_type   : str  = "avg",
    use_conv    : bool = False
) -> tuple[nn.Module, int]:
    # Flatten when we use a Linear layer after pooling
    flatten_in_pool = not use_conv
    if not pool_type:
        if not (num_classes == 0 or use_conv):
            raise ValueError(
                f"Pooling can only be disabled if classifier is also removed "
                f"or conv classifier is used."
            )
        # Disable flattening if pooling is pass-through (no pooling)
        flatten_in_pool = False
    
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type, flatten=flatten_in_pool
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def _create_fc(
    num_features: int, num_classes: int, use_conv: bool = False
) -> nn.Module:
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, (1, 1), bias=True)
    else:
        # NOTE: using my Linear wrapper that fixes AMP + torchscript casting
        # issue
        fc = Linear(num_features, num_classes, bias=True)
    return fc


def create_classifier(
    num_features: int,
    num_classes : int,
    pool_type   : str  = "avg",
    use_conv    : bool = False
) -> tuple[nn.Module, nn.Module]:
    global_pool, num_pooled_features = _create_pool(
        num_features, num_classes, pool_type, use_conv=use_conv
    )
    fc = _create_fc(num_pooled_features, num_classes, use_conv=use_conv)
    return global_pool, fc


# MARK: - Modules

@HEADS.register(name="classifier")
class Classifier(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_type  : str   = "avg",
        drop_rate  : float = 0.0,
        use_conv   : bool  = False
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(
            in_channels, num_classes, pool_type, use_conv=use_conv
        )
        self.fc = _create_fc(
            num_pooled_features, num_classes, use_conv=use_conv
        )
        self.flatten = (
            nn.Flatten(1) if use_conv and pool_type else nn.Identity()
        )

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(
                x, p=float(self.drop_rate), training=self.training
            )
        x = self.fc(x)
        x = self.flatten(x)
        return x


@HEADS.register(name="detector")
class Detector(nn.Module):
    """Detector head."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_classes: int,
        anchors    : Union[list, tuple] = (),
        channels   : Union[list, tuple] = ()
    ):
        super().__init__()
        self.num_classes = num_classes           # Number of classes
        self.num_outputs = num_classes + 5       # Number of outputs per anchor
        self.num_layers  = len(anchors)          # Number of measurement layers
        self.num_anchors = len(anchors[0]) // 2  # Number of anchors
        self.grid        = [torch.zeros(1)] * self.num_layers  # Init grid
        self.stride      = None  # Strides computed during build
        
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer("anchors", a)  # shape(num_layers, num_anchors, 2)
        self.register_buffer(
            "anchor_grid",
            a.clone().view(self.num_layers, 1, -1, 1, 1, 2)
        )  # shape(num_layers, 1, num_anchors, 1, 1, 2)
        self.m = nn.ModuleList(
            nn.Conv2d(in_channels, self.num_outputs * self.num_anchors, (1, 1))
            for in_channels in channels
        )  # output conv
        self.export = False  # onnx export
    
    # MARK: Configure

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        # input = input.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])      # Conv
            bs, _, ny, nx = x[i].shape  # input(bs,255,20,20) to input(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx)
            x[i] = x[i].permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 +
                               self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_outputs))
        
        return x if self.training else (torch.cat(z, 1), x)
