#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for SIREN (Sinusoidal Representation Networks) in Implicit Neural
Representations (INRs).

This module provides implementations of SIREN layers and MLPs using sine
activation functions, which are particularly effective for representing
complex signals and functions.

References:
    - Paper: "Implicit Neural Representations with Periodic Activation Functions,"
      NeurIPS 2020.
    - Code: https://github.com/vsitzmann/siren
"""

__all__ = [
    "DepthAwareSineLayer",
    "SIREN",
    "SineLayer",
    "SineLayerBN",
]

import numpy as np
import torch
import torch.nn as nn

from mon.nn.mlp.linear import DepthAwareLinear


# --- Layer ---
class SineLayer(nn.Module):
    r"""A sine layer.
    
    It applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.

    References:
        - Code: https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        init_weights: bool  = True,
    ):
        """Initializes the SineLayer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first (bool): Flag indicating if this is the first layer.
                Defaults to False.
            omega_0 (float): Frequency scaling factor. Defaults to 30.0.
            init_weights (bool): If True, initializes weights. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                             1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.omega_0,
                                             np.sqrt(6.0 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SineLayer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return torch.sin(self.omega_0 * self.linear(input))


class SineLayerBN(nn.Module):
    r"""A sine layer with batch normalization.
    
    It applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        init_weights: bool  = True,
    ):
        """Initializes the SineLayerBN.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first (bool): Flag indicating if this is the first layer.
                Defaults to False.
            omega_0 (float): Frequency scaling factor. Defaults to 30.0.
            init_weights (bool): If True, initializes weights. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        self.norm        = nn.BatchNorm1d(out_features)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                             1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.omega_0,
                                             np.sqrt(6.0 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SineLayerBN.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return torch.sin(self.norm(self.omega_0 * self.linear(input)))


class DepthAwareSineLayer(nn.Module):
    r"""A depth-aware sine layer.
    
    It applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is
    a frequency factor and :math:`\sin` is the sine function.

    References:
        - Code: https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        depth_features: int,
        kernel_size   : int   = 3,
        alpha         : float = 8.3,
        bias          : bool  = True,
        is_first      : bool  = False,
        omega_0       : float = 30.0,
        init_weights  : bool  = True,
    ):
        """Initializes the DepthAwareSineLayer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            depth_features (int): Number of depth features.
            kernel_size (int): Kernel size for depth-aware linear layer.
                Defaults to 3.
            alpha (float): Scaling factor for depth-aware linear layer.
                Defaults to 8.3.
            bias (bool): If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first (bool): Flag indicating if this is the first layer.
                Defaults to False.
            omega_0 (float): Frequency scaling factor. Defaults to 30.0.
            init_weights (bool): If True, initializes weights. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.dalinear    = DepthAwareLinear(
            in_features    = in_features,
            out_features   = out_features,
            depth_features = depth_features,
            kernel_size    = kernel_size,
            alpha          = alpha,
            bias           = bias,
        )
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                self.dalinear.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features
                )
            else:
                self.dalinear.linear.weight.uniform_(
                    -np.sqrt(6.0 / self.in_features) / self.omega_0,
                     np.sqrt(6.0 / self.in_features) / self.omega_0
                )

    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DepthAwareSineLayer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
            depth (torch.Tensor): Depth tensor of shape (..., depth_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return torch.sin(self.omega_0 * self.dalinear(input, depth))


# --- MLP ---
class SIREN(nn.Module):
    """An implementation of SIREN MLP.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        hidden_dim    : int,
        hidden_layers : int,
        first_omega_0 : float = 30.0,
        hidden_omega_0: float = 30.0,
        bias          : bool  = True,
    ):
        """Initializes the SIREN MLP.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Number of hidden units in each hidden layer.
            hidden_layers (int): Number of hidden layers.
            first_omega_0 (float): Frequency scaling factor for the first layer.
                Defaults to 30.0.
            hidden_omega_0 (float): Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            bias (bool): If True, adds a learnable bias to the linear layers.
        """
        super().__init__()
        # First layer
        self.net = []
        self.net.append(SineLayer(in_features, hidden_dim, bias, is_first=True, omega_0=first_omega_0))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_dim, hidden_dim, bias, is_first=False, omega_0=hidden_omega_0))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                                          np.sqrt(6.0 / hidden_dim) / hidden_omega_0)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SIREN MLP.
        
        Args:
            coords (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        return self.net(coords)
