#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "GAUSS",
    "GaussLayer",
]

import torch
import torch.nn as nn


# ----- Layer -----
class GaussLayer(nn.Module):
    r"""Applies an affine linear transformation with Gaussian activation to the
    incoming data: :math:`y = \exp(-(xA^T + b)^2)`, where :math:`\exp`
    is the exponential function.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        scale: Gaussian scale factor. Default: ``30.0``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        scale       : float = 30.0,
        *args, **kwargs
    ):
        super().__init__()
        self.scale  = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(self.scale * self.linear(input)) ** 2)


# ----- MLP -----
class GAUSS(nn.Module):
    """Implements the Gaussian MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        scale: Gaussian scale factor. Default: ``30.0``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        scale        : float = 30.0,
        bias         : bool  = True,
    ):
        super().__init__()
        # First layer
        self.net = []
        self.net.append(GaussLayer(in_features, hidden_dim, bias, scale=scale))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(GaussLayer(hidden_dim, hidden_dim, bias, scale=scale))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
