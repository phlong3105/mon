#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class data atomic operations.

This module provides pure functions that perform a single mathematical or
structural change to the class data.
"""

__all__ = [
    "class_id_to_one_hot",
]

import numpy as np


# ==============================================================================
# VALIDATION & SANITIZATION (Integrity Checks)
# ==============================================================================

# --- Verify (Schema and range checking) ---


# --- Clean (Fixing corrupt values/nulls) ---


# ==============================================================================
# GEOMETRIC TRANSFORMATIONS (Resizing, Warping)
# ==============================================================================

# --- Analytics (Area, Perimeter, Centroid calculations) ---


# --- Metrics ---


# --- Project (Affine, Perspective, and Coordinate space transforms) ---


# --- Reshape (Resize, Crop, Padding) ---


# ==============================================================================
# STATISTICAL OPERATIONS (Normalization, Scaling)
# ==============================================================================

# --- Normalize (Mean/Std, Min-Max scaling) ---


# --- Standardize (Unit conversion) ---
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
        ValueError: If ``class_id`` is negative.
        ValueError: If ``class_id`` is out of range for ``num_classes``.
    """
    if num_classes <= 0:
        raise ValueError(f"``num_classes`` must be a positive integer, got {num_classes}.")
    if class_id < 0:
        raise ValueError(f"``class_id`` must be a non-negative integer, got {class_id}.")
    if not (0 <= class_id < num_classes):
        raise ValueError(f"``class_id`` is out of range for ``num_classes`` {num_classes}, got {class_id}.")
    
    probs = np.zeros(num_classes, dtype=np.float32)
    probs[class_id] = 1.0
    return probs
