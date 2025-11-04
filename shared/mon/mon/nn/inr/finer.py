#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FINER.

References:
    - Paper: "FINER: Flexible spectral-bias tuning in Implicit NEural
      Representation by Variable-periodic Activation Functions," CVPR 2024.
    - Code: https://github.com/liuzhen0212/FINER
"""

__all__ = [
    "FINER",
    "FINERLayer",
    "FINER_PP",
]

import numpy as np
import torch
import torch.nn as nn


# ----- Layer -----
class FINERLayer(nn.Module):
    r"""Applies an affine linear transformation with scaled sine activation to
    the incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b) \cdot \text{scale})`,
    where :math:`w_0` is a frequency factor and :math:`\sin` is the sine function.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        is_first: First layer flag for initialization. Default: ``False``.
        omega_0: Frequency scaling factor. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer. Default: ``None``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.
        init_weights: Initializes weights if ``True``. Default: ``True``.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        bias            : bool  = True,
        is_first        : bool  = False,
        omega_0         : float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False,
        init_weights    : bool  = True,
    ):
        super().__init__()
        self.in_features      = in_features
        self.is_first         = is_first
        self.omega_0          = omega_0
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad   = scale_req_grad
        self.linear           = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()
        if self.first_bias_scale:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on the layer position in the network."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6.0 / self.in_features) / self.omega_0,
                                             np.sqrt(6.0 / self.in_features) / self.omega_0)
    
    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
    
    def scale(self, linear: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor after linear transformation."""
        if self.scale_req_grad:
            return torch.abs(linear) + 1
        with torch.no_grad():
            return torch.abs(linear) + 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        linear = self.linear(input)
        scale  = self.scale(linear)
        return torch.sin(self.omega_0 * scale * linear)
        
       
# ----- MLP -----
class FINER(nn.Module):
    """Implements the FINER MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        first_omega_0: Frequency scaling factor for the first layer. Default: ``30.0``.
        hidden_omega_0: Frequency scaling factor for the hidden layers. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float`` or ``None``.
            Default: ``None``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.

    References:
        - Paper: "FINER: Flexible spectral-bias tuning in Implicit NEural
          Representation by Variable-periodic Activation Functions," CVPR 2024.
        - Code: https://github.com/liuzhen0212/FINER
    """
    
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False,
        bias            : bool  = True,
    ):
        super().__init__()
        # First layer
        self.net = []
        self.net.append(FINERLayer(in_features, hidden_dim, bias, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(FINERLayer(hidden_dim, hidden_dim, bias, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                                          np.sqrt(6.0 / hidden_dim) / hidden_omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class FINER_PP(nn.Module):
    """Implements the FINER++ MLP.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        hidden_dim: Hidden channel dimensions.
        hidden_layers: Number of hidden layers.
        first_omega_0: Frequency scaling factor for the first layer. Default: ``30.0``.
        hidden_omega_0: Frequency scaling factor for the hidden layers. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float`` or ``None``.
            Default: ``None``.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.

    References:
        - Paper: "FINER++: Building a Family of Variable-periodic Functions for
          Activating Implicit Neural Representation," arXiv 2025.
        - Code: https://github.com/liuzhen0212/FINER
    """
    
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float = 5,
        scale_req_grad  : bool  = False,
        bias            : bool  = True,
    ):
        super().__init__()
        self.out_features = out_features
        
        # First layer
        self.net = []
        self.net.append(FINERLayer(in_features, hidden_dim, bias, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(FINERLayer(hidden_dim, hidden_dim, bias, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                                          np.sqrt(6.0 / hidden_dim) / hidden_omega_0)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        output = self.net(coords)
        return output.view(-1, self.out_features)
