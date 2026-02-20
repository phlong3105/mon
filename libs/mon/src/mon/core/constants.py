#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Constants.

This module defines various global constants used throughout the ``mon`` package.
"""

from __future__ import annotations

__all__ = [
    "ALBUMENTATIONS",
    "BACKBONES",
    "DATASETS",
    "DIRS",
    "EXT",
    "MODELS",
    "MONO_ROOT",
    "ROOT",
    "VERBOSE",
    "WEIGHTS",
    "ZOO_ROOT",
]

from types import SimpleNamespace

from .dtype import ImageExtension, WeightExtension
from .factory import DatasetFactory, Factory, ModelFactory, WeightsFactory
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
# region FACTORIES
# ==============================================================================

ALBUMENTATIONS: Factory = Factory(name="Albumentations", decamelize=False)
DATASETS: DatasetFactory = DatasetFactory(name="Datasets", decamelize=True)
BACKBONES: ModelFactory = ModelFactory(name="Backbones", decamelize=True)
MODELS: ModelFactory = ModelFactory(name="Models", decamelize=True)
WEIGHTS: WeightsFactory = WeightsFactory(name="Weights", decamelize=True)

# endregion


# ==============================================================================
# region VALUES
# ==============================================================================

DIRS = SimpleNamespace(
    DEBUG="debug",
    DEPTH="depth",
    IMAGE="image",
    LABEL="label",
    PRED="pred",
    VISUALIZE="visualize",
)

EXT = SimpleNamespace(
    CKPT=WeightExtension.CKPT,
    IMAGE=ImageExtension.JPG,
    POINTCLOUD=".ply",
    WEIGHTS=WeightExtension.PT,
)

VERBOSE = True  # Global verbosity flag for internal logging

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
