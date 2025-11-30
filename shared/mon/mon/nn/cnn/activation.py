#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for activation functions.

This module provides various activation functions commonly used in neural networks
to introduce non-linearity into the model.
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
    """An activation unit applies simple-gate activation unit by chunking the
    input tensor into two halves and multiplying them element-wise.
    
    References:
        - Paper: https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SimpleGate activation unit.
        
        Args:
            input (torch.Tensor): The input tensor of shape (B, 2*C, ...).
            
        Returns:
            torch.Tensor: The output tensor after applying the SimpleGate activation
                unit, of shape (B, C, ...).
        """
        x1, x2 = input.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """A Sine activation function as described in the SIREN paper.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    def __init__(self, w0: float = 1.0):
        """Initializes the Sine activation function.
        
        Args:
            w0 (float): The frequency scaling factor. Default is 1.0.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Sine activation function.
        
        Args:
            input (torch.Tensor): The input tensor of shape (B, C, ...).
            
        Returns:
            torch.Tensor: The output tensor after applying the Sine activation
                function.
        """
        return torch.sin(self.w0 * input)
