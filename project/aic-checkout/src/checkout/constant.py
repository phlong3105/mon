#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`checkout` package.
"""

from __future__ import annotations

__all__ = [
    "CAMERA",
    "CONTENT_ROOT_DIR",
    "DATA_DIR",
    "DETECTION",
    "DISTANCE",
    "MOTION",
    "OBJECT",
    "RUN_DIR",
    "SOURCE_ROOT_DIR",
    "TRACKING",
    "WEIGHT_DIR",
]

import os

import mon
from mon.constant import *

# region Factory

CAMERA    = mon.Factory(name="Camera")
DETECTION = mon.Factory(name="Detection")
MOTION    = mon.Factory(name="Motion")
OBJECT    = mon.Factory(name="Object")
TRACKING  = mon.Factory(name="Tracking")

# endregion


# region Directory

__current_file   = mon.Path(__file__).absolute()      # "aic-checkout/src/checkout/constant.py"
SOURCE_ROOT_DIR  = __current_file.parents[1]          # "aic-checkout/src/"
CONTENT_ROOT_DIR = __current_file.parents[2]          # "aic-checkout"
WEIGHT_DIR       = CONTENT_ROOT_DIR / "weight"        # "aic-checkout/weight"
RUN_DIR          = mon.Path()       / "run"
DATA_DIR         = os.getenv("DATA_DIR", None)        # If we've set value in the os.environ
if DATA_DIR is None:
    DATA_DIR = mon.Path("/data")                      # Run from Docker container
else:
    DATA_DIR = mon.Path(DATA_DIR)
if not DATA_DIR.is_dir():
    DATA_DIR = CONTENT_ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = ""

# endregion
