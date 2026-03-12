#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Constants.

This module defines various global constants used throughout the ``mon`` package.
"""

from __future__ import annotations

__all__ = [
    "K",
]

from types import SimpleNamespace

from .dtype import ImageExtension, WeightExtension
from .path import Path


# ==============================================================================
# region PATHS
# ==============================================================================

# Find the ``mon`` package root (first pyproject.toml up)
current_file = Path(__file__).normalize()
root = current_file
for parent in current_file.parents:
    if (parent / "pyproject.toml").exists():
        root = parent
        break

# Find the monorepo root (highest pyproject.toml up)
all_roots = [p for p in root.parents if (p / "pyproject.toml").exists()]
mono_root = all_roots[-1] if all_roots else root

# Find the model zoo directory (prefer root-level zoo if it exists)
zoo_root = root / "zoo"
if not zoo_root.exists():
    zoo_root = mono_root / "zoo"

# endregion


# ==============================================================================
# region VALUES
# ==============================================================================

class K(SimpleNamespace):
    """Class for constants."""

    # --- Paths ---
    ROOT = root
    MONO_ROOT = mono_root
    ZOO_ROOT = zoo_root

    # --- Directories ---
    DEBUG_DIR = "debug"
    DEPTH_DIR = "depth"
    IMAGE_DIR = "image"
    LABEL_DIR = "label"
    PRED_DIR = "pred"
    VIS_DIR = "vis"

    # --- Extensions ---
    CKPT_EXT = WeightExtension.CKPT
    IMAGE_EXT = ImageExtension.JPG
    WEIGHTS_EXT = WeightExtension.PT

    # --- Logging ---
    VERBOSE = True

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
