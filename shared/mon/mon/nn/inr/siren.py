#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SIREN.

References:
    - Paper: "Implicit Neural Representations with Periodic Activation Functions," NeurIPS 2020.
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
from mon.core.nn.modules.linear import DepthAwareLinear


# ----- Layer -----
class SineLayer(nn.Module):
    r"""Applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        is_first: First layer flag for weight initialization. Default: ``False``.
        omega_0: Frequency scaling factor. Default: ``30.0``.
        init_weights: Initializes weights if ``True``. Default: ``True``.
        kwargs: Additional keyword arguments for ``torch.nn.Linear``.

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
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the network."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                             1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.omega_0,
                                             np.sqrt(6.0 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class SineLayerBN(nn.Module):
    """Applies an affine linear transformation with sine activation and
    batch normalization.
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
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        self.norm        = nn.BatchNorm1d(out_features)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                             1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.omega_0,
                                             np.sqrt(6.0 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.norm(self.omega_0 * self.linear(input)))


class DepthAwareSineLayer(nn.Module):
    r"""Applies an affine linear transformation with sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        is_first: First layer flag for weight initialization. Default: ``False``.
        omega_0: Frequency scaling factor. Default: ``30.0``.
        init_weights: Initializes weights if ``True``. Default: ``True``.
        
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
        """Initializes linear layer weights based on the layer position in the network."""
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
        return torch.sin(self.omega_0 * self.dalinear(input, depth))


# ----- MLP -----
class SIREN(nn.Module):
    """Implements the SIREN MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        first_omega_0: Frequency scaling factor for the first layer. Default: ``30.0``.
        hidden_omega_0: Frequency scaling factor for the hidden layers. Default: ``30.0``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
    
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
        return self.net(coords)
