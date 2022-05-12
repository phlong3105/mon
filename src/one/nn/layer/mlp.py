#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MLP module w/ dropout and configurable activation layer.
"""

from __future__ import annotations

from typing import Optional

from torch import nn as nn
from torch import Tensor

from one.core import Callable
from one.core import MLP_LAYERS
from one.core import to_2tuple

__all__ = [
    "ConvMlp",
    "GatedMlp",
    "GluMlp",
    "Mlp",
]


# MARK: - Modules

@MLP_LAYERS.register(name="conv_mlp")
class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_features    : int,
        hidden_features: Optional[int]      = None,
        out_features   : Optional[int]      = None,
        act_layer      : Callable           = nn.ReLU,
        norm_layer     : Optional[Callable] = None,
        drop           : float              = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(
            in_features, hidden_features, kernel_size=(1, 1),
            bias=True
            )
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=(1, 1),
            bias=True
            )
        self.drop = nn.Dropout(drop)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@MLP_LAYERS.register(name="glu_mlp")
class GluMlp(nn.Module):
    """MLP w/ GLU style gating. See: https://arxiv.org/abs/1612.08083,
    https://arxiv.org/abs/2002.05202
    """

    # MARK: Magic Functions

    def __init__(
        self,
        in_features    : int,
        hidden_features: Optional[int] = None,
        out_features   : Optional[int] = None,
        act_layer      : Callable      = nn.Sigmoid,
        drop           : float         = 0.0
    ):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        if hidden_features % 2 != 0:
            raise ValueError
        drop_probs = to_2tuple(drop)

        self.fc1   = nn.Linear(in_features, hidden_features)
        self.act   = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2   = nn.Linear(hidden_features // 2, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    # MARK: Configure

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x        = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x        = x * self.act(gates)
        x        = self.drop1(x)
        x        = self.fc2(x)
        x        = self.drop2(x)
        return x


@MLP_LAYERS.register(name="gated_mlp")
class GatedMlp(nn.Module):
    """MLP as used in gMLP."""

    # MARK: Magic Functions

    def __init__(
        self,
        in_features    : int,
        hidden_features: Optional[int]      = None,
        out_features   : Optional[int]      = None,
        act_layer      : Callable           = nn.GELU,
        gate_layer     : Optional[Callable] = None,
        drop           : float              = 0.0
    ):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        drop_probs      = to_2tuple(drop)

        self.fc1   = nn.Linear(in_features, hidden_features)
        self.act   = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            if hidden_features % 2 != 0:
                raise ValueError
            self.gate = gate_layer(hidden_features)
            # FIXME base reduction on gate property?
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.fc2   = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


@MLP_LAYERS.register(name="mlp")
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_features    : int,
        hidden_features: Optional[int] = None,
        out_features   : Optional[int] = None,
        act_layer      : Callable      = nn.GELU,
        drop           : float         = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
