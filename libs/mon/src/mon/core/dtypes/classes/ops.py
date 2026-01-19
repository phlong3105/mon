#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes atomic operations.

This module provides atomic operations for classes.
"""

from __future__ import annotations

__all__ = [
    "class_id_to_one_hot",
    "class_ids_to_one_hot",
]

import numpy as np


# ==============================================================================
# region CREATION
# ==============================================================================


# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================


# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region MUTATION
# ==============================================================================

# --- Alternation ---


# --- Rearrangement ---


# --- Addition ---


# --- Removal ---


# endregion


# ==============================================================================
# region COMPUTATION
# ==============================================================================

# --- Arithmetic ---


# --- Comparison ---


# --- Logical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---

def class_id_to_one_hot(class_id: int, num_classes: int) -> np.ndarray:
    """Convert a class ID to a one-hot encoded probability array.

    Args:
        class_id: Class ID.
        num_classes: Total number of classes.

    Returns:
        An one-hot encoded probability array of shape (``num_classes``) where
        the index corresponding to ``class_id`` is 1.0 and all other indices
        are 0.0.

    Raises:
        ValueError: If ``num_classes`` is not a positive integer.
        ValueError: If ``class_id`` is not in range [0, ``num_classes``).
    """
    # Validation
    if num_classes <= 0:
        raise ValueError(f"Expected 'num_classes' to be a positive integer, but got {num_classes}.")
    if not (0 <= class_id < num_classes):
        raise ValueError(f"Expected 'class_id' in range [0, {num_classes}), but got {class_id}.")

    probs = np.zeros(num_classes, dtype=np.float32)
    probs[class_id] = 1.0
    return probs


def class_ids_to_one_hot(class_ids: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a batch of class IDs to one-hot encoded probability arrays.

    Args:
        class_ids: Array of class IDs of shape (N,).
        num_classes: Total number of classes.

    Returns:
        An array of shape (N, ``num_classes``) where each row is a one-hot
        encoded probability array corresponding to the class ID in ``class_ids``.

    Raises:
        ValueError: If ``num_classes`` is not a positive integer.
        ValueError: If any value in ``class_ids`` is not in range [0, ``num_classes``).
    """
    # Ensure class_ids is a 1D array
    class_ids = np.atleast_1d(class_ids)

    # Validation
    if num_classes <= 0:
        raise ValueError(f"Expected 'num_classes' to be a positive integer, but got {num_classes}.")
    if class_ids.size > 0:
        if class_ids.min() < 0 or class_ids.max() >= num_classes:
            raise ValueError(f"Expected all 'class_ids' in range [0, {num_classes}).")

    # The Identity Matrix trick: indexing into np.eye(N) returns the one-hot vectors
    return np.eye(num_classes, dtype=np.float32)[class_ids]


# --- Standardization ---


# --- Structural ---


# --- Statistical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region DESTRUCTION
# ==============================================================================


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
