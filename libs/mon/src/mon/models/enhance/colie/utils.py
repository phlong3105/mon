#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "get_coords",
    "get_patches",
    "replace_v_component",
]

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Color ---

def replace_v_component(image_hsv: Tensor, v_new: Tensor) -> Tensor:
    """Replaces the V component of an HSV image (1, 3, H, W)."""
    image_hsv[:, -1] = v_new
    return image_hsv


# --- Features ---

def get_coords(h: int, w: int) -> Tensor:
    """Creates a coordinates grid for INF."""
    coords = np.dstack(
        np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w)
        )
    )
    coords = torch.from_numpy(coords).float()
    return coords


def get_patches(image: Tensor, kernel_size: int) -> Tensor:
    """Creates a tensor where the channel contains patch information."""
    kernel = torch.zeros(
        (kernel_size ** 2, 1, kernel_size, kernel_size)
    ).to(image.device)

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
