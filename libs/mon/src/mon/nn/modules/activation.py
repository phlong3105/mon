#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation Layers.

This module provides activation layers.
"""

from __future__ import annotations

__all__ = [
    "SimpleGate",
    "Sine",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region LAYERS
# ==============================================================================

class SimpleGate(nn.Module):
    """Simple-gate activation unit.

    Chunk the input tensor into two halves along the channel dimension and
    multiply them element-wise. Use this parameter-free activation function in
    modern architectures like NAFNet.

    References:
        - Paper: https://arxiv.org/pdf/2204.04676.pdf
    """

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, ...) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, ...) and values ranging from
                0.0 to 1.0.
        """
        x1, x2 = x.chunk(chunks=2, dim=1)
        return x1 * x2


class Sine(nn.Module):
    """Sine activation function as described in the SIREN paper.

    Apply a sine transformation to the input, scaled by a frequency factor
    ``w0``. Use this in implicit neural representations.

    References:
        - Code: https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, w0: float = 1.0):
        """Initialize a new instance.

        Args:
            w0 (float, optional): Frequency factor. Defaults to 1.0.
        """
        super().__init__()
        self.w0 = w0

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from -1.0 to 1.0.
        """
        return torch.sin(self.w0 * x)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
