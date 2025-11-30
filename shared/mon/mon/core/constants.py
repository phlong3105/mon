#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for global constants.

This module defines global constants used across the project, including
directory paths, file extensions, and configuration flags.
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


# ----- Directory -----
current_file = Path(__file__).absolute()   # mon/shared/mon/mon/constants.py
ROOT_DIR     = current_file.parents[3]     # ./mon
ZOO_DIR      = ROOT_DIR / "zoo"            # ./mon/zoo


# ----- Constants -----
DEPTH_SOURCE       = DepthSource.DAv2_ViTB
INFRARED_SOURCE    = InfraredSource.INFRARED
SAVE_DEBUG_DIR     = "debug"
SAVE_IMAGE_DIR     = "pred"
SAVE_LABEL_DIR     = "label"
SAVE_VISUALIZE_DIR = "visualize"
SAVE_CKPT_EXT      = WeightExtension.CKPT.value
SAVE_IMAGE_EXT     = ImageExtension.JPG.value
SAVE_WEIGHTS_EXT   = WeightExtension.PT.value
VERBOSE            = True  # Global verbosity flag for internal logging
