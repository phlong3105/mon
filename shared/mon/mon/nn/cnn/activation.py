#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements activation layers."""

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
    """Applies simple-gate activation unit by chunking the input tensor into two
    halves and multiplying them element-wise.
    
    Shape:
        - Input: :math:`(B, C, H, W)`.
        - Output: :math:`(B, C/2, H, W)`.
        
    References:
        - Paper: https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1, x2 = input.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """Applies the sine activation unit function element-wise.

    Args:
        w0: The frequency scaling factor. Default: ``1.0``.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * input)
    
    def extra_repr(self) -> str:
        return f"w0={self.w0}"
