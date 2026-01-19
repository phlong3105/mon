#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Project-wide constant definitions.

This module provides project-wide constants and default directory and file extensions.
"""

from __future__ import annotations

__all__ = [
    "DIRS",
    "EXT",
    "MONO_ROOT_DIR",
    "ROOT_DIR",
    "SOURCE",
    "VERBOSE",
    "ZOO_DIR",
]

from types import SimpleNamespace

from mon.core.enum import (
    DepthSource,
    ImageExtension,
    InfraredSource,
    WeightExtension,
)
from mon.core.pathlib import Path


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# --- Paths ---

# Robustly find the project root (first pyproject.toml up)
current_file = Path(__file__).normalize()
ROOT_DIR     = current_file

for parent in current_file.parents:
    if (parent / "pyproject.toml").exists():
        ROOT_DIR = parent
        break

# Find the monorepo root (highest pyproject.toml up)
_all_roots    = [p for p in ROOT_DIR.parents if (p / "pyproject.toml").exists()]
MONO_ROOT_DIR = _all_roots[-1] if _all_roots else ROOT_DIR

# Zoo directory (prefer root-level zoo if it exists)
_zoo_dir_in_root = ROOT_DIR / "zoo"

if _zoo_dir_in_root.exists():
    ZOO_DIR = _zoo_dir_in_root
else:
    ZOO_DIR = MONO_ROOT_DIR / "zoo"


# --- Values ---

DIRS = SimpleNamespace(
    DEBUG     = "debug",
    DEPTH     = "depth",
    IMAGE     = "image",
    LABEL     = "label",
    PRED      = "pred",
    VISUALIZE = "visualize",
)

EXT = SimpleNamespace(
    CKPT       = WeightExtension.CKPT.value,
    IMAGE      = ImageExtension.JPG.value,
    POINTCLOUD = ".ply",
    WEIGHTS    = WeightExtension.PT.value,
)

SOURCE = SimpleNamespace(
    DEPTH    = DepthSource.DAv2_ViTB,
    INFRARED = InfraredSource.INFRARED,
)

# --- Execution Flags  ---

VERBOSE = True  # Global verbosity flag for internal logging

# endregion
