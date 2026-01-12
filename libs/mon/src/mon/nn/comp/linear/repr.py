#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Representation-based linear layers.

This module provides various linear layers used for representation learning.
"""

from __future__ import annotations

__all__ = [
    "ComplexGaborLayer",
    "DepthAwareSineLinear",
    "FINERLinear",
    "GaussLinear",
    "RealGaborLayer",
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
        in_features (int): Size of each input sample.
        is_first (bool): Flag indicating if this is the first layer.
        omega_0 (float): Frequency scaling factor.
        linear (torch.nn.Linear): The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        init_weights: bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 30.0.
            init_weights: If True, initializes weights. Defaults to True.
        """
        super().__init__()
        self.in_features = in_features
        self.is_first    = is_first
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.in_features,
                     1.0 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6.0 / self.in_features) / self.omega_0,
                     np.sqrt(6.0 / self.in_features) / self.omega_0
                )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SineLinearBN(nn.Module):
    """Sine linear layer with batch normalization.

    Apply an affine linear transformation with sine activation and batch
    normalization to the incoming data.

    Attributes:
        in_features (int): Size of each input sample.
        is_first (bool): Flag indicating if this is the first layer.
        omega_0 (float): Frequency scaling factor.
        linear (torch.nn.Linear): The underlying linear layer.
        norm (torch.nn.BatchNorm1d): The batch normalization layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        is_first    : bool  = False,
        omega_0     : float = 30.0,
        init_weights: bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 30.0.
            init_weights: If True, initializes weights. Defaults to True.
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
        """Initialize linear layer weights based on the layer position in the
        network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.in_features,
                     1.0 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6.0 / self.in_features) / self.omega_0,
                     np.sqrt(6.0 / self.in_features) / self.omega_0
                )

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return torch.sin(self.norm(self.omega_0 * self.linear(x)))


# --- Geometry-Informed Units ---

class DepthAwareSineLinear(nn.Module):
    """Depth-aware sine linear layer.

    Apply a depth-aware linear transformation with sine activation to the
    incoming data.

    Attributes:
        in_features (int): Size of each input sample.
        is_first (bool): Flag indicating if this is the first layer.
        omega_0 (float): Frequency scaling factor.
        dalinear (DepthAwareLinear): The underlying depth-aware linear layer.
    """

    # --- Lifecycle & Initialization ---
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
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            depth_features: Number of depth features.
            kernel_size: Kernel size for depth-aware linear layer. Defaults to 3.
            alpha: Scaling factor for depth-aware linear layer. Defaults to 8.3.
            bias: If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: Flag indicating if this is the first layer. Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 30.0.
            init_weights: If True, initializes weights. Defaults to True.
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
        """Initialize linear layer weights based on the layer position in the
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

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.
            d: Depth tensor of shape (..., depth_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return torch.sin(self.omega_0 * self.dalinear(x, d))


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
        in_features (int): Size of each input sample.
        is_first (bool): If True, initializes weights for the first layer.
        omega_0 (float): Frequency scaling factor.
        first_bias_scale (float | None): Bias scale for the first layer.
        scale_req_grad (bool): Scale requires gradient if True.
        linear (torch.nn.Linear): The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        bias            : bool  = True,
        is_first        : bool  = False,
        omega_0         : float = 30.0,
        first_bias_scale: float | None = None,
        scale_req_grad  : bool  = False,
        init_weights    : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: If True, initializes weights for the first layer.
                Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 30.0.
            first_bias_scale: Bias scale for the first layer as float or None.
                Defaults to None.
            scale_req_grad: Scale requires gradient if True. Defaults to False.
            init_weights: If True, initializes the weights of the linear layer.
                Defaults to True.
        """
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
        """Initialize weights for the linear layer."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6.0 / self.in_features) / self.omega_0,
                     np.sqrt(6.0 / self.in_features) / self.omega_0
                )

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

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        linear = self.linear(x)
        scale  = self.scale(linear)
        return torch.sin(self.omega_0 * scale * linear)

# endregion


# ==============================================================================
# region GAUSSIAN LINEAR LAYERS
# ==============================================================================

class GaussLinear(nn.Module):
    """Gaussian linear layer.

    Apply a Gaussian activation to the output of a linear transformation.

    Attributes:
        scale (float): Gaussian scale factor.
        linear (torch.nn.Linear): The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool  = True,
        scale       : float = 30.0,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
                Defaults to True.
            scale: Gaussian scale factor. Defaults to 30.0.
        """
        super().__init__()
        self.scale  = scale
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


# ==============================================================================
# WIRE (WAVELET IMPLICIT NEURAL REPRESENTATIONS)
# ==============================================================================

class RealGaborLayer(nn.Module):
    r"""Real Gabor layer.

    Apply an affine linear transformation with real Gabor activation to the
    incoming data.

    Apply the transformation: :math:`y = \cos(w_0 \cdot (xA^T + b)) \cdot
    \exp(-(\text{scale} \cdot (xA^T + b))^2)`, where :math:`w_0` is a
    frequency factor, :math:`\cos` is the cosine function, and :math:`\exp` is
    the exponential function.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        omega_0 (float): Frequency scaling factor.
        scale_0 (float): Scaling of Gabor Gaussian term.
        is_first (bool): First layer flag for initialization.
        in_features (int): Size of each input sample.
        freqs (torch.nn.Linear): Linear layer for frequency component.
        scale (torch.nn.Linear): Linear layer for scale component.
    """

    # --- Lifecycle & Initialization ---
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
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: First layer flag for initialization. Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 10.0.
            sigma_0: Scaling of Gabor Gaussian term. Defaults to 10.0.
            trainable: If True, omega_0 and sigma_0 are trainable parameters.
                Defaults to False.
        """
        super().__init__()
        self.omega_0     = omega_0
        self.scale_0     = sigma_0
        self.is_first    = is_first
        self.in_features = in_features
        self.freqs       = nn.Linear(in_features, out_features, bias=bias)
        self.scale       = nn.Linear(in_features, out_features, bias=bias)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        return torch.cos(omega) * torch.exp(-(scale ** 2))


class ComplexGaborLayer(nn.Module):
    r"""Complex Gabor layer.

    Apply an affine linear transformation with complex Gabor activation to the
    incoming data.

    Apply the transformation: :math:`y = \exp(i \cdot w_0 \cdot (xA^T + b))
    \cdot \exp(-(\text{scale} \cdot (xA^T + b))^2)`, where :math:`w_0` is a
    frequency factor, :math:`i` is the imaginary unit, and :math:`\exp` is
    the exponential function.

    Attributes:
        omega_0 (torch.nn.Parameter | float): Frequency scaling factor.
        scale_0 (torch.nn.Parameter | float): Scaling of Gabor Gaussian term.
        is_first (bool): First layer flag for initialization.
        in_features (int): Size of each input sample.
        linear (torch.nn.Linear): The underlying linear layer.
    """

    # --- Lifecycle & Initialization ---
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
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If False, the layer will not learn an additive bias.
                Defaults to True.
            is_first: First layer flag for initialization. Defaults to False.
            omega_0: Frequency scaling factor. Defaults to 10.0.
            sigma_0: Scaling of Gabor Gaussian term. Defaults to 40.0.
            trainable: If True, omega_0 and sigma_0 are trainable parameters.
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
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        self.linear  = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        lin   = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())

# endregion
