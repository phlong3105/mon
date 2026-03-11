#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Linear Layers.

This package contains various linear layers commonly used in MLP and deep
neural networks.
"""

from __future__ import annotations

__all__ = [
    "FINERLinear",
    "LinearTime",
    "SineLinear",
    "SineLinearBN",
    "SineLinearTime",
]

import numpy as np
import torch
from torch import nn, Tensor


# ==============================================================================
# region LAYERS
# ==============================================================================

# --- Linear Layers ---


# --- Periodic Units ---

class SineLinear(nn.Module):
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
        w0: float = 30.0,
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

        # Define layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if not self.is_last:
            self.init_weights()

    def init_weights(self):
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
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
        return x if self.is_last else torch.sin(self.w0 * x)


class SineLinearBN(nn.Module):
    """Sine linear layer with batch normalization.

    Apply an affine linear transformation with sine activation and batch
    normalization to the incoming data.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30.0,
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

        # Define layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        if not self.is_last:
            self.init_weights()

    def init_weights(self):
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
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
        return self.norm(x) if self.is_last else torch.sin(self.norm(self.w0 * x))


# --- Spectral Tuning ---

class FINERLinear(nn.Module):
    r"""FINER linear layer.

    Apply an affine linear transformation with scaled sine activation to the
    incoming data: :math:`y = \sin(w_0 \cdot (xA^T + b) \cdot \text{scale})`,
    where :math:`w_0` is a frequency factor and :math:`\sin` is the sine function.

    References:
        - Paper: "FINER: Flexible spectral-bias tuning in Implicit NEural
          Representation by Variable-periodic Activation Functions," CVPR 2024.
        - Code: https://github.com/liuzhen0212/FINER
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad: bool = False,
        is_first: bool = False,
        is_last: bool = False,
        bias: bool = True,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            w0 (float, optional): Frequency scaling factor. Defaults to 30.0.
            first_bias_scale (float, optional): Scaling factor for the first
                layer bias. Defaults to None.
            scale_req_grad (bool, optional): Flag indicating whether the scaling
                factor requires gradient computation. Defaults to False.
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
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad
        self.is_first = is_first
        self.is_last = is_last

        # Define layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if not self.is_last:
            self.init_weights()
        if self.first_bias_scale:
            self.init_first_bias()

    def init_weights(self):
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                b = 1.0 / self.in_features
            else:
                b = np.sqrt(6.0 / self.in_features) / self.w0
            with torch.no_grad():
                self.linear.weight.uniform_(-b, b)

    def init_first_bias(self):
        """Initialize bias for the first layer."""
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def scale(self, linear: Tensor) -> Tensor:
        """Generate the scaling factor after linear transformation.

        Args:
            linear (Tensor): Output of the linear transformation.

        Returns:
            Tensor: Scaling factor.
        """
        if self.scale_req_grad:
            return torch.abs(linear) + 1
        with torch.no_grad():
            return torch.abs(linear) + 1

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
        x = self.scale(x)
        return x if self.is_last else torch.sin(self.w0 * x)


# --- Time-Conditioned Linear Layers ---

class LinearTime(nn.Linear):
    """Linear layer that takes in the time step as an additional input."""

    # --- Lifecycle & Initialization ---
    def __init__(self, in_features: int, *args, **kwargs):
        """Initialize a new instance.

        Args:
            in_features (int): Number of features in the input (excluding the
                time feature).
        """
        super().__init__(in_features=in_features + 1, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input feature tensor of shape (B, N, F) and values
                ranging from 0.0 to 1.0.
        """
        t_feat = torch.ones_like(x[:, :, :1]) * t  # (B, N, 1)
        t_and_x = torch.cat([t_feat, x], dim=-1)  # (B, N, F + 1)
        return super(LinearTime, self).forward(t_and_x)


class SineLinearTime(SineLinear):
    r"""Sine linear layer with periodic activation and time step."""

    # --- Lifecycle & Initialization ---
    def __init__(self, in_features: int, *args, **kwargs):
        """Initialize a new instance.

        Args:
            in_features (int): Number of features in the input (excluding the
                time feature).
        """
        super().__init__(in_features + 1, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input tensor of shape (..., in_features) and values
                ranging from 0.0 to 1.0.
        """
        t_feat = torch.ones_like(x[:, :, :1]) * t  # (B, N, 1)
        t_and_x = torch.cat([t_feat, x], dim=-1)  # (B, N, F + 1)
        return super(SineLinearTime, self).forward(t_and_x)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
