#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Positional Encoding Layers.

This module provides various positional encoding (PE) layers used for
representing high-dimensional inputs.
"""

from __future__ import annotations

__all__ = [
    "FourierPE",
]

import math

import torch
from torch import nn, Tensor


# ==============================================================================
# region LAYERS
# ==============================================================================

class FourierPE(nn.Module):
    """Positional Encoding (PE) using Fourier."""

    # --- Lifecycle & Initialization ---
    def __init__(self, mapping_size: int, B: float = 20.0):
        """Initialize a new instance.

        Args:
            mapping_size (int): Size of the output feature space.
            B (float, optional): Scaling factor for the Fourier features.
                Defaults to 20.0.
        """
        super().__init__()
        self.in_features = mapping_size // 2
        self.out_features = mapping_size

        if B is None:
            self.B = None
        else:
            self.register_buffer("B", torch.randn((self.in_features, 2)) * B)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the layer.

        Args:
            x (Tensor): Input tensor of shape (..., in_features) and values
                ranging from -1.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from -1.0 to 1.0.
        """
        if self.B is None:
            return x
        else:
            proj = (2.0 * math.pi * x) @ self.B.T
            encoding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
            return encoding

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
