#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IZ-DCE utilities.

This module provides various utilities for IZ-DCE.
"""

from __future__ import annotations

__all__ = [
    "get_coords",
    "weights_init",
]

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.core import Size, SizeLike
from mon.cv.ops import FastGuidedFilter


# ==============================================================================
# region UTILITIES
# ==============================================================================

def get_coords(size: SizeLike, device: torch.device) -> Tensor:
    """Create a normalized square coordinates grid.

    Args:
        size (SizeLike): Size of the grid.
        device (torch.device): Device to use for computation.

    Returns:
        Tensor: Coordinates tensor of shape (1, H, W, 2) and values ranging from
            -1.0 to 1.0.
    """
    size = Size.from_value(size)
    h, w = size.hw

    # 1. Create linear spaces for Height and Width
    # We use 'ij' indexing: h_coords corresponds to rows, w_coords to columns
    h_coords = torch.linspace(-1, 1, steps=h, device=device)
    w_coords = torch.linspace(-1, 1, steps=w, device=device)

    # 2. Generate the meshgrid
    # indexing='ij' ensures that grid[i, j] corresponds to (h_i, w_j)
    # This prevents the 90-degree rotation issues.
    grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

    # 3. Stack and Reshape
    # The last dimension must be (x, y). In grid_sample terms:
    # index 0 is the horizontal (W) axis, index 1 is the vertical (H) axis.
    grid = torch.stack([grid_w, grid_h], dim=-1) # Shape: [H, W, 2]

    # 4. Flatten for the MLP
    # Shape becomes [1, H*W, 2] to support batch processing
    grid = grid.view(1, -1, 2)

    return grid


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
