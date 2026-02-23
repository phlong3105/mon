#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "ReflectanceINR",
    "ResidualINR",
    "SirenLayer",
]

import numpy as np
import torch
from torch import nn, Tensor


class SirenLayer(nn.Module):
    r"""Sine linear layer.

    Apply an affine linear transformation with sine activation to the incoming
    data: :math:`y = \sin(w_0 \cdot (xA^T + b))`, where :math:`w_0` is a
    frequency factor and :math:`\sin` is the sine function.

    References:
        - Paper: "Implicit Neural Representations with Periodic Activation
          Functions," NeurIPS 2020.
        - Code: https://github.com/vsitzmann/siren
        - Code: https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30,
        is_first: bool = False,
        is_last: bool = False,
        bias: bool = True,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            w0 (float, optional): Frequency scaling factor. Defaults to 30.0.
            is_first (bool, optional): Flag indicating if this is the first layer.
                Defaults to False.
            is_last (bool, optional): Flag indicating if this is the last layer.
                Defaults to False.
           bias (bool, optional): If True, adds a learnable bias to the linear
                layers. Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.in_features = in_features
        self.out_features = out_features
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last

        # Define layers
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if not self.is_last:
            self.init_weights()

    def init_weights(self):
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        if self.is_first:
            b = 1.0 / self.in_features
        else:
            b = np.sqrt(6.0 / self.in_features) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Apply an affine linear transformation with sine activation to the
        incoming data. If this is the last layer, no activation is applied.

        Args:
            x (Tensor): Input tensor of shape (..., in_features) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from 0.0 to 1.0.
        """
        x = self.linear(x)
        return nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)


class ResidualINR(nn.Module):

    def __init__(
        self,
        patch_dim: int,
        num_layers: int,
        hidden_dim: int,
        add_layer: int,
        weight_decay: list[float] | None = None
    ):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''
        patch_layers = [SirenLayer(patch_dim, hidden_dim, is_first=True)]
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        output_layers = []

        for _ in range(1, add_layer - 2):
            patch_layers.append(SirenLayer(hidden_dim, hidden_dim))
            spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))
        patch_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(SirenLayer(hidden_dim, hidden_dim // 2))

        for _ in range(add_layer, num_layers - 1):
            output_layers.append(SirenLayer(hidden_dim, hidden_dim))
        output_layers.append(SirenLayer(hidden_dim, 1, is_last=True))

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]

        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),"weight_decay": weight_decay[2]}]

    def forward(self, patch: Tensor, spatial: Tensor) -> Tensor:
        return self.output_net(
            torch.cat(
                (self.patch_net(patch), self.spatial_net(spatial)), -1
            )
        )


class ReflectanceINR(nn.Module):

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        weight_decay: list[float] | None = None
    ):
        super().__init__()
        '''
        `add_layer` should be in range of  [1, num_layers-2]
        '''
        spatial_layers = [SirenLayer(2, hidden_dim, is_first=True)]
        for _ in range(1, num_layers - 1):
            spatial_layers.append(SirenLayer(hidden_dim, hidden_dim))
        spatial_layers.append(SirenLayer(hidden_dim, 1, is_last=True))

        self.spatial_net = nn.Sequential(*spatial_layers)

        if not weight_decay:
            weight_decay = [0.1]

        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]

    def forward(self, spatial: Tensor) -> Tensor:
        return self.spatial_net(spatial)
