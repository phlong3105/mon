#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Representation-based linear layers.

This module provides various linear layers used for representation learning.
"""

from __future__ import annotations

__all__ = [
    "DepthAwareSineLinear",
    "FINERLinear",
    "SineLinear",
    "SineLinearBN",
]

import numpy as np
import torch
import torch.nn as nn

from .depth_aware import DepthAwareLinear


# ==============================================================================
# region SINE LINEAR LAYERS
# ==============================================================================

# --- Basic Periodic Units ---

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

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        w0: Frequency scaling factor.
        is_first: Flag indicating if this is the first layer.
        is_last: Flag indicating if this is the last layer.
        linear: The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features : int,
        out_features: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        is_last     : bool  = False,
        bias        : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            w0: Frequency scaling factor. Defaults to 30.0.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            is_last: Flag indicating if this is the last layer. Defaults to False.
            bias: If False, the layer will not learn an additive bias. Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.in_features  = in_features
        self.out_features = out_features
        self.w0           = w0
        self.is_first     = is_first
        self.is_last      = is_last

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Apply an affine linear transformation with sine activation to the
        incoming data. If this is the last layer, no activation is applied.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SineLinearBN(nn.Module):
    """Sine linear layer with batch normalization.

    Apply an affine linear transformation with sine activation and batch
    normalization to the incoming data.

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        w0: Frequency scaling factor.
        is_first: Flag indicating if this is the first layer.
        is_last: Flag indicating if this is the last layer.
        linear: The underlying linear layer.
        norm: The batch normalization layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features : int,
        out_features: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        is_last     : bool  = False,
        bias        : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            w0: Frequency scaling factor. Defaults to 30.0.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            is_last: Flag indicating if this is the last layer. Defaults to False.
            bias: If False, the layer will not learn an additive bias. Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.in_features  = in_features
        self.out_features = out_features
        self.w0           = w0
        self.is_first     = is_first
        self.is_last      = is_last

        # Define layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm   = nn.BatchNorm1d(out_features)
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Apply an affine linear transformation with sine activation to the
        incoming data. If this is the last layer, no activation is applied.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        x = self.linear(x)
        return self.norm(x) if self.is_last else torch.sin(self.norm(self.w0 * x))


# --- Geometry-Informed Units ---

class DepthAwareSineLinear(nn.Module):
    """Depth-aware sine linear layer.

    Apply a depth-aware linear transformation with sine activation to the
    incoming data.

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        w0: Frequency scaling factor.
        is_first: Flag indicating if this is the first layer.
        is_lastFlag indicating if this is the last layer.
        da_linear: The underlying depth-aware linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        depth_features: int,
        kernel_size   : int   = 3,
        alpha         : float = 8.3,
        w0            : float = 30.0,
        is_first      : bool  = False,
        is_last       : bool  = False,
        bias          : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            depth_features: Number of depth features.
            kernel_size: Kernel size for depth-aware linear layer. Defaults to 3.
            alpha: Scaling factor for depth-aware linear layer. Defaults to 8.3.
            w0: Frequency scaling factor. Defaults to 30.0.
            bias: If False, the layer will not learn an additive bias. Defaults to True.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            is_last: Flag indicating if this is the last layer. Defaults to False.
            bias: If False, the layer will not learn an additive bias. Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.in_features  = in_features
        self.out_features = out_features
        self.is_first     = is_first
        self.w0           = w0
        self.is_first     = is_first
        self.is_last      = is_last

        # Define layer
        self.da_linear = DepthAwareLinear(
            in_features    = in_features,
            out_features   = out_features,
            depth_features = depth_features,
            kernel_size    = kernel_size,
            alpha          = alpha,
            bias           = bias,
        )
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
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Apply an affine linear transformation with sine activation to the
        incoming data. If this is the last layer, no activation is applied.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.
            d: Depth tensor of shape (..., depth_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        x = self.da_linear(x, d)
        return x if self.is_last else torch.sin(self.w0 * x)


# --- Advanced Spectral Tuning ---

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

    Attributes:
        in_features: Size of each input sample.
        is_first: If True, initializes weights for the first layer.
        w0: Frequency scaling factor.
        first_bias_scale: Bias scale for the first layer.
        scale_req_grad: Scale requires gradient if True.
        is_first: Flag indicating if this is the first layer.
        is_last: Flag indicating if this is the last layer.
        linear: The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        w0              : float = 30.0,
        first_bias_scale: float = None,
        scale_req_grad  : bool  = False,
        is_first        : bool  = False,
        is_last         : bool  = False,
        bias            : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            w0: Frequency scaling factor. Defaults to 30.0.
            first_bias_scale: Bias scale for the first layer as float or None.
                Defaults to None.
            scale_req_grad: Scale requires gradient if True. Defaults to False.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            is_last: Flag indicating if this is the last layer. Defaults to False.
            bias: If set to False, the layer will not learn an additive bias.
                Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.in_features      = in_features
        self.out_features     = out_features
        self.w0               = w0
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad   = scale_req_grad
        self.is_first         = is_first
        self.is_last          = is_last

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

    def scale(self, linear: torch.Tensor) -> torch.Tensor:
        """Generate the scaling factor after linear transformation.

        Args:
            linear: The output of the linear transformation.

        Returns:
            The scaling factor.
        """
        if self.scale_req_grad:
            return torch.abs(linear) + 1
        with torch.no_grad():
            return torch.abs(linear) + 1

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Apply an affine linear transformation with sine activation to the
        incoming data. If this is the last layer, no activation is applied.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        x = self.linear(x)
        x = self.scale(x)
        return x if self.is_last else torch.sin(self.w0 * x)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
