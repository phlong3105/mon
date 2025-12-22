#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Gaussian MLP architecture.

This module implements the Gaussian MLP architecture for Implicit Neural
Representation (INR) using Gaussian activation functions.
"""

__all__ = [
    "GAUSS",
    "GaussLayer",
]

import torch
import torch.nn as nn


# --- Layer ---
class GaussLayer(nn.Module):
    r"""An implementation of a Gaussian layer.
    
    It applies an affine linear transformation with Gaussian activation to the
    incoming data: :math:`y = \exp(-(xA^T + b)^2)`, where :math:`\exp` is the
    exponential function.
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        scale       : float = 30.0,
    ):
        """Initializes the Gaussian layer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If set to False, the layer will not learn an additive
                bias. Defaults to True.
            scale (float): Gaussian scale factor. Defaults to 30.0.
        """
        super().__init__()
        self.scale  = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Gaussian layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return torch.exp(-(self.scale * self.linear(input)) ** 2)


# --- MLP ---
class GAUSS(nn.Module):
    """An implementation of a Gaussian MLP.

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
        """Initializes the Gaussian MLP.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Number of hidden units in each hidden layer.
            hidden_layers (int): Number of hidden layers.
            scale (float): Gaussian scale factor. Defaults to 30.0.
            bias (bool): If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
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
        """Forward pass of the Gaussian MLP.
        
        Args:
            coords (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return self.net(coords)
