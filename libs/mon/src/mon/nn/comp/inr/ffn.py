#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Fourier-Feature Networks (FFN).

This module implements Fourier-Feature Networks (FFN) for Implicit Neural
Representation (INR).

References:
    - Code: https://xeonqq.github.io/machine%20learning/fourier-feature-siren/
"""

__all__ = [
    "FFEncoding",
    "FFEncodingMLP",
]

import numpy as np
import torch
import torch.nn as nn


# --- Layer ---
class FFEncoding(nn.Module):
    """An implementation of Fourier Feature Encoding layer."""
    
    def __init__(self, in_features: int, B: float = 20.0):
        """Initializes the Fourier Feature Encoding layer.
        
        Args:
            in_features (int): Size of each input sample.
            B (float): Standard deviation of the Gaussian distribution used to
                sample the projection matrix. If set to None, no projection is
                applied. Defaults to 20.0.
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = in_features * 2
        if B is None:
            self.B = None
        else:
            self.register_buffer("B", torch.randn((in_features, 2)) * B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier Feature Encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
        
        Returns:
            torch.Tensor: Encoded tensor of shape (..., out_features).
        """
        if self.B is None:
            return x
        else:
            x_proj    = (2. * np.pi * x) @ self.B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding
        

# --- MLP ---
class FFEncodingMLP(nn.Module):
    """An implementation of Fourier Feature Encoding MLP.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        B            : float = 20.0,
        bias         : bool  = True,
    ):
        """Initializes the Fourier Feature Encoding MLP.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Number of hidden units in each hidden layer.
            hidden_layers (int): Number of hidden layers.
            B (float): Standard deviation of the Gaussian distribution used to
                sample the projection matrix. If set to None, no projection is
                applied. Defaults to 20.0.
            bias (bool): If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
        super().__init__()
        self.encoding = FFEncoding(in_features=in_features, B=B)
        
        # First layer
        self.net = []
        self.net.append(nn.Linear(self.encoding.out_features, hidden_dim, bias=bias))
        self.net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier Feature Encoding MLP.
        
        Args:
            coords (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return self.net(self.encoding(coords))
