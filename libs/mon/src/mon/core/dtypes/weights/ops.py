#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model weights atomic operations.

This module provides atomic operations for model weights.
"""

from __future__ import annotations

__all__ = [
    "resolve_weights",
    "resolve_weights_dir",
    "resolve_weights_file",
]

from mon.core.constants import ZOO_DIR
from mon.core.pathlib import Path
from mon.core.utils import is_valid_str
from .core import Weights


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

def resolve_weights_dir(root: Path | str, weights: Path | str) -> Path | None:
    """Resolve the weight directory from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.

    Returns:
        Absolute weights directory path or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Ensure weights is always a Path object
    weights = Path(weights) if is_valid_str(weights) else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_dir():
        return weights

    # Check Local Project Root (Highest Priority)
    local_dir = root / weights
    if local_dir.is_dir():
        return local_dir

    # Check Global Zoo Directory
    global_dir = ZOO_DIR / weights
    if global_dir.is_dir():
        return global_dir

    # Return None if not found
    return None


def resolve_weights_file(root: Path | str, weights: Path | str) -> Path | None:
    """Resolve the weight file from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.

    Returns:
        Absolute weight file path or None if nothing was found.
    """
    root = Path(root).normalize(exist=True)
    # Ensure weights is always a Path object
    weights = Path(weights) if is_valid_str(weights) else None

    # Check if the weight provided is already an absolute path
    if weights.is_absolute() and weights.is_weights_file():
        return weights

    # Check Local Project Root (Highest Priority)
    # Search specifically for the file in the project's training runs
    local_file = root / weights
    if local_file.is_weights_file(exist=True):
        return local_file

    # Check Global Zoo Directory
    global_file = ZOO_DIR / weights
    if global_file.is_weights_file(exist=True):
        return weights

    # Return None if not found
    return None


def resolve_weights(
    root       : Path | str,
    weights    : Path | str,
    num_classes: int | None = None,
) -> Weights | None:
    """Resolve a ``Weights`` object from the given root and weights name or
    relative path.

    Args:
        root: Project root path.
        weights: Weights name or relative path.
        num_classes: Optional number of classes to set in the Weights object.
            Defaults to None.

    Returns:
        ``Weights`` object if found, otherwise None.
    """
    from mon.core.factory import WEIGHTS

    weights = resolve_weights_file(root=root, weights=weights)

    # If a valid weights file was found, wrap it in a Weights object
    if weights:
        # Check if the weights object is already registered in WEIGHTS
        if WEIGHTS.has(path=weights):
            return WEIGHTS.find_weights_objs(path=weights)
        # Otherwise, the weights object has not been registered yet.
        else:
            return Weights(path=weights, num_classes=num_classes)

    # Return None if not found
    return None


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


# --- Standardization ---


# --- Structural ---


# --- Statistical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region DESTRUCTION
# ==============================================================================


# endregion
