#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities Operations.

This module provides general-purpose utility operations.
"""

from __future__ import annotations

__all__ = [
    "normalize_min_max",
]

from mon.core import TensorOrArray


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---

def normalize_min_max(image: TensorOrArray, eps: float = 1e-8) -> TensorOrArray:
    """Stretch image values to the range [0.0, 1.0].

    Args:
        image (TensorOrArray): Image, formatted as a tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0; or as an array of shape (H, W, C)
            and values ranging from 0 to 255.
        eps (float): Small value to prevent division by zero when the image has
            constant pixel values. Defaults to 1e-8.

    Returns:
        TensorOrArray: Normalized image of the same shape and type as the input,
            but with values stretched to the range [0.0, 1.0].
    """
    return (image - image.min()) / (image.max() - image.min() + eps)


# --- Structural ---


# --- Statistical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
