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
    "MovingState",
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


# region Enum

class MovingState(mon.Enum):
    """The counting state of an object when moving through the camera."""
    Candidate   = 1  # Preliminary state.
    Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
    Counting    = 3  # Object is in the counting zone/counting state.
    ToBeCounted = 4  # Mark object to be counted somewhere in this loop iteration.
    Counted     = 5  # Mark object has been counted.
    Exiting     = 6  # Mark object for exiting the ROI or image frame. Let's it die by itself.

# endregion
