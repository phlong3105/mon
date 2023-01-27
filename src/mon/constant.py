#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`mon` package."""

from __future__ import annotations

__all__ = [
    # Extend :mod:`mon.core.constant`
    "CONTENT_ROOT_DIR", "DATA_DIR", "DOCS_DIR", "FILE_HANDLER", "ImageFormat",
    "MemoryUnit", "PROJECT_DIR", "RUN_DIR", "SNIPPET_DIR", "SOURCE_ROOT_DIR",
    "VideoFormat", "WEIGHT_DIR",
    # Extend :mod:`mon.coreimage.constant`
    "AppleRGB", "BBoxFormat", "BasicRGB", "BorderType", "CFA", "DISTANCE",
    "DistanceMetric", "IMG_MEAN", "IMG_STD", "InterpolationMode", "PaddingMode",
    "RGB", "VISION_BACKEND", "VisionBackend",
    # Extend :mod:`mon.coreml.constant`
    "ACCELERATOR", "CALLBACK", "DATAMODULE", "DATASET", "LAYER", "LOGGER",
    "LOSS", "LR_SCHEDULER", "MODEL", "METRIC", "ModelPhase", "OPTIMIZER",
    "Reduction", "STRATEGY", "TRANSFORM",
]

from mon.core.constant import *
from mon.coreimage.constant import *
from mon.coreml.constant import *
