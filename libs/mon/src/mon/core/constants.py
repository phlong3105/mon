#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Project-wide constant definitions.

This module provides project-wide constants and default directory and file
extensions for configuration and I/O operations.
"""

__all__ = [
    "DEPTH_SOURCE",
    "INFRARED_SOURCE",
    "ROOT_DIR",
    "SAVE_CKPT_EXT",
    "SAVE_DEBUG_DIR",
    "SAVE_IMAGE_DIR",
    "SAVE_IMAGE_EXT",
    "SAVE_LABEL_DIR",
    "SAVE_VISUALIZE_DIR",
    "SAVE_WEIGHTS_EXT",
    "VERBOSE",
    "ZOO_DIR",
]

from mon.core.enum import (
    DepthSource,
    ImageExtension,
    InfraredSource,
    WeightExtension,
)
from mon.core.pathlib import Path


# ==============================================================================
# PATH ORCHESTRATION
# ==============================================================================

# --- Roots (Calculating the absolute base of the project) ---
current_file = Path(__file__).absolute()   # mon/shared/mon/mon/constants.py
ROOT_DIR     = current_file.parents[4]     # ./mon


# --- Resources ---
ZOO_DIR = ROOT_DIR / "zoo"                 # ./mon/zoo


# ==============================================================================
# IO & PERSISTENCE DEFAULTS
# ==============================================================================

# --- Directory Names (Standard folder names for outputs) ---
SAVE_DEBUG_DIR     = "debug"
SAVE_IMAGE_DIR     = "pred"
SAVE_LABEL_DIR     = "label"
SAVE_VISUALIZE_DIR = "visualize"


# --- Extensions (Allowed/Default file formats) ---
SAVE_CKPT_EXT    = WeightExtension.CKPT.value
SAVE_IMAGE_EXT   = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT = WeightExtension.PT.value


# ==============================================================================
# IO & PERSISTENCE DEFAULTS
# ==============================================================================

# --- Execution Flags (Verbosity, debug modes) ---
VERBOSE = True  # Global verbosity flag for internal logging


# --- Algorithm Defaults (Source selection, model types) ---
DEPTH_SOURCE    = DepthSource.DAv2_ViTB
INFRARED_SOURCE = InfraredSource.INFRARED
