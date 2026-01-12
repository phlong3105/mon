#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation functions.

This module provides various activation layers used for introducing
non-linearity into neural networks.
"""

from __future__ import annotations

__all__ = [
    "CELU",
    "ELU",
    "GELU",
    "GLU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "MultiheadAttention",
    "PReLU",
    "RReLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "SiLU",
    "Sigmoid",
    "SimpleGate",
    "Sine",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
]

import torch
import torch.nn as nn
from torch.nn.modules.activation import (
    CELU,
    ELU,
    GELU,
    GLU,
    Hardshrink,
    Hardsigmoid,
    Hardswish,
    Hardtanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    MultiheadAttention,
    PReLU,
    RReLU,
    ReLU,
    ReLU6,
    SELU,
    SiLU,
    Sigmoid,
    Softmax,
    Softmax2d,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Tanh,
    Tanhshrink,
    Threshold,
)


class SimpleGate(nn.Module):
    """Simple-gate activation unit.

    Chunk the input tensor into two halves along the channel dimension and
    multiply them element-wise. Use this parameter-free activation function in
    modern architectures like NAFNet.

    References:
        - Paper: https://arxiv.org/pdf/2204.04676.pdf
    """
    
    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, 2*C, ...) and values ranging from
                0.0 to 1.0.

        Returns:
            Output tensor of shape (B, C, ...) and values ranging from 0.0
            to 1.0.
        """
        x1, x2 = x.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """Sine activation function as described in the SIREN paper.

    Apply a sine transformation to the input, scaled by a frequency factor
    ``w0``. Use this in implicit neural representations.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py

    Attributes:
        w0 (float): The frequency scaling factor.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, w0: float = 1.0):
        """Initialize a new instance.

        Args:
            w0: The frequency scaling factor. Defaults to 1.0.
        """
        super().__init__()
        self.w0 = w0

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through the layer.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging from
                0.0 to 1.0.

        Returns:
            Output tensor of shape (B, C, H, W) and values ranging from 0.0
            to 1.0.
        """
        return torch.sin(self.w0 * x)
