#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for RetinexNet.
"""

from __future__ import annotations

__all__ = [
    "ave_gradient",
    "gradient",
    "smooth",
]

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

def gradient(x: Tensor, direction: Literal["x", "y"]) -> Tensor:
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(x.device)
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    grad = torch.abs(F.conv2d(x, kernel, stride=1, padding=1))
    return grad

def ave_gradient(x: Tensor, direction: Literal["x", "y"]) -> Tensor:
    return F.avg_pool2d(gradient(x, direction), kernel_size=3, stride=1, padding=1)

def smooth(R: Tensor, L: Tensor) -> Tensor:
    R = 0.299 * R[:, 0, :, :] + 0.587 * R[:, 1, :, :] + 0.114 * R[:, 2, :, :]
    R = torch.unsqueeze(R, dim=1)
    return torch.mean(
        gradient(L, "x") * torch.exp(-10 * ave_gradient(R, "x")) +
        gradient(L, "y") * torch.exp(-10 * ave_gradient(R, "y"))
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
