#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for WIRE MLP with Gabor wavelet activations.

This module implements the WIRE (Wavelet Implicit Neural Representations) MLP
architecture using Gabor wavelet activations. It includes both real and complex
Gabor layers, as well as the WIRE MLP class itself.

References:
    - Paper: "WIRE: Wavelet Implicit Neural Representations," CVPR 2023.
    - Code: https://github.com/vishwa91/wire
"""

__all__ = [
    "ComplexGaborLayer",
    "RealGaborLayer",
    "WIRE",
]

import numpy as np
import torch
import torch.nn as nn


# --- Layer ---
class RealGaborLayer(nn.Module):
    r"""A layer that applies an affine linear transformation with real Gabor
    activation to the incoming data.
    
    It applies the transformation: :math:`y = \cos(w_0 \cdot (xA^T + b)) \cdot
    \exp(-(\text{scale} \cdot (xA^T + b))^2)`, where :math:`w_0` is a
    frequency factor, :math:`\cos` is the cosine function, and :math:`\exp` is
    the exponential function.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 10.0,
        sigma_0     : float = 10.0,
        trainable   : bool  = False
    ):
        """Initializes the RealGaborLayer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first (bool): First layer flag for initialization. Defaults to False.
            omega_0 (float): Frequency scaling factor. Defaults to 10.0.
            sigma_0 (float): Scaling of Gabor Gaussian term. Defaults to 10.0.
            trainable (bool): If True, omega_0 and sigma_0 are trainable parameters.
                Defaults to False.
        """
        super().__init__()
        self.omega_0     = omega_0
        self.scale_0     = sigma_0
        self.is_first    = is_first
        self.in_features = in_features
        self.freqs       = nn.Linear(in_features, out_features, bias=bias)
        self.scale       = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RealGaborLayer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        return torch.cos(omega) * torch.exp(-(scale ** 2))


class ComplexGaborLayer(nn.Module):
    r"""A layer that applies an affine linear transformation with complex Gabor
    activation to the incoming data.
    
    It applies the transformation: :math:`y = \exp(i \cdot w_0 \cdot (xA^T + b))
    \cdot \exp(-(\text{scale} \cdot (xA^T + b))^2)`, where :math:`w_0` is a
    frequency factor, :math:`i` is the imaginary unit, and :math:`\exp` is
    the exponential function.
    """
    
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 10.0,
        sigma_0     : float = 40.0,
        trainable   : bool  = False
    ):
        """Initializes the ComplexGaborLayer.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first (bool): First layer flag for initialization. Defaults to False.
            omega_0 (float): Frequency scaling factor. Defaults to 10.0.
            sigma_0 (float): Scaling of Gabor Gaussian term. Defaults to 40.0.
            trainable (bool): If True, omega_0 and sigma_0 are trainable parameters.
                Defaults to False.
        """
        super().__init__()
        self.omega_0     = omega_0
        self.scale_0     = sigma_0
        self.is_first    = is_first
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
        
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        self.linear  = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ComplexGaborLayer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        lin   = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


# --- MLP ---
class WIRE(nn.Module):
    """A WIRE MLP with Gabor wavelet activations.
    
    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        hidden_dim    : int,
        hidden_layers : int   = 4,
        first_omega_0 : float = 10.0,
        hidden_omega_0: float = 10.0,
        scale         : float = 10.0,
        bias          : bool  = True,
    ):
        """Initializes the WIRE MLP.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Size of each hidden layer.
            hidden_layers (int): Number of hidden layers. Defaults to 4.
            first_omega_0 (float): Frequency scaling factor for the first layer.
                Defaults to 10.0.
            hidden_omega_0 (float): Frequency scaling factor for hidden layers.
                Defaults to 10.0.
            scale (float): Scaling of Gabor Gaussian term. Defaults to 10.0.
            bias (bool): If False, the layers will not learn an additive bias.
                Defaults to True.
        """
        super().__init__()
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin  = ComplexGaborLayer
        
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_dim   = int(hidden_dim / np.sqrt(2))
        dtype        = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"
        
        # Legacy parameter
        self.pos_encode = False
        
        # First layer
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_dim, bias, is_first=True, omega_0=first_omega_0, sigma_0=scale, trainable=False))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_dim, hidden_dim, bias, is_first=False, omega_0=hidden_omega_0, sigma_0=scale))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias, dtype=dtype)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the WIRE MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        output = self.net(x)
        if self.wavelet == "gabor":
            return output.real
        return output
