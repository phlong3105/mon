#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation functions.

This module implements various activation layers used for introducing
non-linearity into neural networks.
"""

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
from torch.nn.modules.activation import *


class SimpleGate(nn.Module):
    """Simple-gate activation unit.
     
     Chunk the input tensor into two halves and multiplying them element-wise.
    
    References:
        - Paper: https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (B, 2*C, ...) and values
                ranging from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (B, C, ...) and values ranging
                from 0.0 to 1.0.
        """
        x1, x2 = input.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """Sine activation function as described in the SIREN paper.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    def __init__(self, w0: float = 1.0):
        """Initialize a new instance.
        
        Args:
            w0: The frequency scaling factor. Defaults to 1.0.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
            
        Returns:
            Output tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return torch.sin(self.w0 * input)
