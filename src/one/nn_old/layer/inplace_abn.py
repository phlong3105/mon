#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inplace Activated Batch Normalization.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn as nn
from torch import Tensor

from one.core import Callable
from one.core import NORM_LAYERS

try:
    from inplace_abn.functions import inplace_abn, inplace_abn_sync
    has_iabn = True
except ImportError:
    has_iabn = False

    def inplace_abn(
        input           : Tensor,
        weight          : Tensor,
        bias            : Tensor,
        running_mean    : Tensor,
        running_var     : Tensor,
        training        : bool     = True,
        momentum        : float    = 0.1,
        eps             : float    = 1e-05,
        activation      : Callable = "leaky_relu",
        activation_param: float    = 0.01
    ):
        raise ImportError("Please install InplaceABN:"
                          "'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.12'")

    def inplace_abn_sync(**kwargs):
        inplace_abn(**kwargs)


__all__ = [
    "inplace_abn",
    "inplace_abn_sync",
    "InplaceAbn"
]


# MARK: - InplaceAbn

@NORM_LAYERS.register(name="inplace_abn")
class InplaceAbn(nn.Module):
    """Activated Batch Normalization. This gathers a BatchNorm and an
    activation function in a single module.

    Args:
        num_features (int):
            Number of feature channels in the input and output.
        eps (float):
            Small constant to prevent numerical issues.
        momentum (float):
            Momentum factor applied to compute running statistics.
        affine (bool):
            If `True` apply learned scale and shift transformation after
            normalization.
        act_layer (str, nn.Module, type):
            Name or type of the activation functions. One of:
            [`leaky_relu`, `elu`]. Default: `leaky_relu`.
        act_param (float):
            Negative slope for the `leaky_relu` activation.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        num_features: int,
        eps         : float              = 1e-5,
        momentum    : float              = 0.1,
        affine      : bool               = True,
        apply_act   : bool               = True,
        act_layer   : Callable           = "leaky_relu",
        act_param   : float              = 0.01,
        drop_block  : Optional[Callable] = None
    ):
        super().__init__()
        self.num_features = num_features
        self.affine       = affine
        self.eps          = eps
        self.momentum     = momentum

        if apply_act:
            if isinstance(act_layer, str):
                if act_layer not in ("leaky_relu", "elu", "identity", ""):
                    raise ValueError
                self.act_name = act_layer if act_layer else "identity"
            else:
                # Convert act layer passed as type to string
                if act_layer == nn.ELU:
                    self.act_name = "elu"
                elif act_layer == nn.LeakyReLU:
                    self.act_name = "leaky_relu"
                elif act_layer == nn.Identity:
                    self.act_name = "identity"
                elif True:
                    raise ValueError(
                        f"Invalid act layer {act_layer.__name__} for IABN."
                    )
        else:
            self.act_name = "identity"
        self.act_param = act_param
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var",  torch.ones(num_features))
        self.reset_parameters()

    # MARK: Configure

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var,  1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias,   0)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        yhat = inplace_abn(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.training, self.momentum, self.eps, self.act_name,
            self.act_param
        )
        if isinstance(yhat, tuple):
            yhat = yhat[0]
        return yhat
