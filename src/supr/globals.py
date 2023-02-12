#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`checkout` package.
"""

from __future__ import annotations

__all__ = [
    "CAMERAS",
    "CONTENT_ROOT_DIR",
    "DATA_DIR",
    "DETECTORS",
    "DISTANCES",
    "MOTIONS",
    "MovingState",
    "OBJECTS",
    "RUN_DIR",
    "SOURCE_ROOT_DIR",
    "TRACKERS",
    "WEIGHT_DIR",
]

import os
from typing import TYPE_CHECKING

import mon
from mon.globals import *

if TYPE_CHECKING:
    pass

# region Factory

CAMERAS   = mon.Factory(name="Cameras")
DETECTORS = mon.Factory(name="Detectors")
MOTIONS   = mon.Factory(name="Motions")
OBJECTS   = mon.Factory(name="Objects")
TRACKERS  = mon.Factory(name="Trackers")

# endregion


# region Directory

__current_file = mon.Path(__file__).absolute()
SOURCE_ROOT_DIR  = __current_file.parents[1]
CONTENT_ROOT_DIR = __current_file.parents[2]

WEIGHT_DIR = SOURCE_ROOT_DIR / "weight"
if not WEIGHT_DIR.is_dir():
    WEIGHT_DIR = SOURCE_ROOT_DIR / "weight"
    
RUN_DIR = mon.Path() / "run"
DATA_DIR = os.getenv("DATA_DIR", None)
if DATA_DIR is None:
    DATA_DIR = mon.Path("/data")
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
    CANDIDATE     = 1  # Preliminary state.
    CONFIRMED     = 2  # Confirmed the Detection is a road_objects eligible for counting.
    COUNTING      = 3  # Object is in the counting zone/counting state.
    TO_BE_COUNTED = 4  # Mark object to be counted somewhere in this loop iteration.
    COUNTED       = 5  # Mark object has been counted.
    EXITING       = 6  # Mark object for exiting the ROI or image frame. Let's it die by itself.
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enums."""
        return {
            "candidate"    : cls.CANDIDATE,
            "confirmed"    : cls.CONFIRMED,
            "counting"     : cls.COUNTING,
            "to_be_counted": cls.TO_BE_COUNTED,
            "counted"      : cls.COUNTED,
            "existing"     : cls.EXITING,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enums."""
        return {
            0: cls.CANDIDATE,
            1: cls.CONFIRMED,
            2: cls.COUNTING,
            3: cls.TO_BE_COUNTED,
            4: cls.COUNTED,
            5: cls.EXITING,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MovingState:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> MovingState:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: MovingState | dict | str) -> MovingState | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, MovingState):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None

# endregion
