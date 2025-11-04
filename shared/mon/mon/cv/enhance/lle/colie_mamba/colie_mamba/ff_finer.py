#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "FF_FINER",
]

from typing import Any

import numpy as np
import torch

from mon.core import nn


# ----- FF-FINER -----
class FINERLayer(nn.Module):
    """Applies scaled sine activation to linear transformation.

    Args:
        in_features: Number of input channels as ``int``.
        out_features: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default: ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float``. Default: ``20.0``.
        is_first: First layer flag for initialization as ``bool``. Default: ``False``.
        bias: Uses bias in linear layer if ``True``. Default: ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default: ``False``.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        is_last         : bool  = False,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.w0               = w0
        self.is_first         = is_first
        self.is_last          = is_last
        self.in_features      = in_features
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        self.linear           = nn.Linear(in_features, out_features, bias=bias)
        if not self.is_last:
            self.init_weights()
        if self.first_bias_scale and self.is_first:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_features if self.is_first else np.sqrt(6 / self.in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor for activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Scaling tensor as ``torch.Tensor``.
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with scaled sine activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return nn.Sigmoid()(linear) if self.is_last else torch.sin(self.w0 * scale * linear)
    
    
class FF_FINER(nn.Module):

    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int,
        num_layers  : int,
        add_layer   : int,
        weight_decay: Any = None
    ):
        super().__init__()
        
        self.register_buffer("B1", torch.randn((hidden_dim, 2))         * 20)
        self.register_buffer("B2", torch.randn((hidden_dim, patch_dim)) * 20)

        patch_layers   = [FINERLayer(hidden_dim * 2, hidden_dim, is_first=True)]
        spatial_layers = [FINERLayer(hidden_dim * 2, hidden_dim, is_first=True)]
        output_layers  = []

        for _ in range(1, add_layer - 2):
            patch_layers.append(FINERLayer(hidden_dim, hidden_dim))
            spatial_layers.append(FINERLayer(hidden_dim, hidden_dim))
        patch_layers.append(FINERLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(FINERLayer(hidden_dim, hidden_dim // 2))

        for _ in range(add_layer, num_layers - 1):
            output_layers.append(FINERLayer(hidden_dim, hidden_dim))
        output_layers.append(FINERLayer(hidden_dim, 1, is_last=True))

        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]

        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]

    def forward(self, patch: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        spatial = self.ff_embedding(spatial, self.B1)
        patch   = self.ff_embedding(patch,   self.B2)
        return self.output_net(torch.cat((self.patch_net(patch), self.spatial_net(spatial)), -1))

    def ff_embedding(self, p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        if B is None:
            return p
        else:
            x_proj    = (2. * np.pi * p) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding
