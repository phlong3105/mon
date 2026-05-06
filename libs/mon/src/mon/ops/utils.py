#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities Operations.

This module provides general-purpose utility operations.
"""

from __future__ import annotations

__all__ = [
    "normalize_minmax",
]

from mon.core import TensorOrArray


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---

def normalize_minmax(
    value: TensorOrArray,
    scale: float = 1.0,
    eps: float = 1e-8
) -> TensorOrArray:
    """Stretch image values to the range [0.0, 1.0].

    Args:
        value (TensorOrArray): Value tensor or array to normalize.
        scale (float, optional): Optional scaling factor to apply to the input
            values before normalization. Defaults to 1.0 (no scaling).
        eps (float, optional): Small value to prevent division by zero when the
            image has constant pixel values. Defaults to 1e-8.

    Returns:
        TensorOrArray: Normalized value tensor or array with values ranging
            from 0.0 to 1.0.
    """
    v = value * scale
    return (v - v.min()) / (v.max() - v.min() + eps)


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
