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
ROOT = current_file
for parent in current_file.parents:
    if (parent / "pyproject.toml").exists():
        ROOT = parent
        break

# Find the monorepo root (highest pyproject.toml up)
_all_roots = [p for p in ROOT.parents if (p / "pyproject.toml").exists()]
MONO_ROOT = _all_roots[-1] if _all_roots else ROOT

# Find the model zoo directory (prefer root-level zoo if it exists)
_zoo_dir = ROOT / "zoo"
if _zoo_dir.exists():
    ZOO_ROOT = _zoo_dir
else:
    ZOO_ROOT = MONO_ROOT / "zoo"

# endregion


# ==============================================================================
# region VALUES
# ==============================================================================

class K(SimpleNamespace):
    """Class for constants."""

    # --- Directories ---
    ROOT = ROOT
    MONO_ROOT = MONO_ROOT
    ZOO_ROOT = ZOO_ROOT

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
