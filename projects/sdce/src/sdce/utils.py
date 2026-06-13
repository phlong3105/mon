#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for SDCE.
"""

from __future__ import annotations

__all__ = [
    "get_coords",
    "weights_init",
]

import torch
from torch import Tensor

from mon.core import Size


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Coordinate ---

def get_coords(features: Tensor, size: Size) -> Tensor:
    """Get the normalized coordinate grid for the target resolution.

    Args:
        features (Tensor): Input features of shape (B, C, H, W). This is used
            to determine the batch size B and device.
        size (Size): Target resolution (H, W).

    Returns:
        Tensor: Normalized coordinate grid of shape (B, H*W, 2), where the last
            dimension contains (x, y) coordinates in the range [-1, 1].
    """
    b = features.shape[0]
    device = features.device

    # We map the massive target resolution to the [-1, 1] continuous space.
    h_coords = torch.linspace(-1, 1, steps=size.h, device=device)
    w_coords = torch.linspace(-1, 1, steps=size.w, device=device)
    grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
    # coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).repeat(b, 1, 1)  # [B, H*W, 2]
    # FIX: View as 1 batch, then expand/repeat to match actual batch size B
    coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).expand(b, -1, -1)

    return coords


# --- Weights Initialization ---

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
