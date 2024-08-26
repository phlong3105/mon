#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SIREN Layers.

This module implements SIREN layers. The concept of SIREN is introduced in the
paper: "Implicit Neural Representations with Periodic Activation Functions," by
Sitzmann et al. (2020).

References:
    https://github.com/lucidrains/siren-pytorch
"""

from __future__ import annotations

__all__ = [
	"SIREN",
]

import math

import torch
from torch import nn

from mon.nn.modules import activation as act


# region SIREN

class SIREN(nn.Module):
    """SIREN Layer.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        omega_0: The frequency of the sine activation function.
            Defaults: ``1.0``.
        C: The constant factor for the weight initialization. Defaults: ``6.0``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        is_last: Whether this is the last layer. Defaults: ``False``.
        use_bias: Whether to use bias. Defaults: ``True``.
        activation: The activation function.
            - If :obj:`activation` is ``None``, then :obj:`nn.Sine(omega_0)` is
                used.
            - If :obj:`is_last` is ``True`` and :obj:`activation` is ``None``,
                then :obj:`nn.Sigmoid()` is used.
            Defaults: ``None``.
        dropout: The dropout rate. Defaults: ``0.0``.
    
    References:
        https://github.com/lucidrains/siren-pytorch
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        omega_0     : int   = 1.0,
        C           : float = 6.0,
        is_first    : bool  = False,
        is_last     : bool  = False,
        use_bias    : bool  = True,
        activation  : nn.Module = None,
        dropout     : float = 0.0
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.omega_0      = omega_0
        self.C 		      = C
        self.is_first     = is_first
        self.is_last      = is_last
        self.use_bias     = use_bias
        self.linear       = nn.Linear(self.in_channels, self.out_channels)
        if self.is_last:
            self.activation = act.Sigmoid() if activation is None else activation
        else:
            self.activation = act.Sine(self.omega_0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        if not self.is_last:
            self.init_weights()
    
    def init_weights(self):
        if self.is_first:
            w_std = 1 / self.in_channels
        else:
            w_std = math.sqrt(self.C / self.in_channels) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-w_std, w_std)
            if self.use_bias:
                self.linear.bias.uniform_(-w_std, w_std)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.linear(x)
        y = self.activation(y)
        y = self.dropout(y)
        return y

# endregion
