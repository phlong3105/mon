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
from typing import TYPE_CHECKING

import mon
from mon.constant import *

if TYPE_CHECKING:
    from supr.typing import MovingStateType


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
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enums."""
        return {
            "candidate"    : cls.Candidate,
            "confirmed"    : cls.Confirmed,
            "counting"     : cls.Counting,
            "to_be_counted": cls.ToBeCounted,
            "counted"      : cls.Counted,
            "existing"     : cls.Exiting,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enums."""
        return {
            0: cls.Candidate,
            1: cls.Confirmed,
            2: cls.Counting,
            3: cls.ToBeCounted,
            4: cls.Counted,
            5: cls.Exiting,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MovingState:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> MovingState:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: MovingStateType) -> MovingState | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ImageFormat):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None
    
# endregion
