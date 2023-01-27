#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`mon.foundation`
package.
"""

from __future__ import annotations

__all__ = [
    "CONTENT_ROOT_DIR", "DATA_DIR", "DOCS_DIR", "FILE_HANDLER", "ImageFormat",
    "MemoryUnit", "PROJECT_DIR", "RUN_DIR", "SNIPPET_DIR", "SOURCE_ROOT_DIR",
    "VideoFormat", "WEIGHT_DIR",
]


import os
from typing import TYPE_CHECKING

from mon.core import enum, factory, pathlib

if TYPE_CHECKING:
    from mon.core.typing import (
        ImageFormatType, MemoryUnitType, VideoFormatType,
    )

# region Directory 

__current_file   = pathlib.Path(__file__).absolute()  # "mon/src/mon/core/constant.py"
SOURCE_ROOT_DIR  = __current_file.parents[1]          # "mon/src/"
CONTENT_ROOT_DIR = __current_file.parents[3]          # "mon"
DOCS_DIR         = CONTENT_ROOT_DIR / "docs"          # "mon/docs"
SNIPPET_DIR      = CONTENT_ROOT_DIR / "snippet"       # "mon/snippet"
PROJECT_DIR      = CONTENT_ROOT_DIR / "project"       # "mon/project"
WEIGHT_DIR       = CONTENT_ROOT_DIR / "weight"        # "mon/weight"
RUN_DIR          = pathlib.Path()   / "run"
DATA_DIR         = os.getenv("DATA_DIR", None)        # If we've set value in the os.environ
if DATA_DIR is None:                                  
    DATA_DIR = pathlib.Path("/data")                  # Run from Docker container
else:                                                 
    DATA_DIR = pathlib.Path(DATA_DIR)                 
if not DATA_DIR.is_dir():                             
    DATA_DIR = CONTENT_ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = ""

# endregion


# region Factory

FILE_HANDLER = factory.Factory(name="FileHandler")

# endregion


# region enum.Enum

class ImageFormat(enum.Enum):
    """Image file formats."""
    
    ARW  = ".arw"
    BMP  = ".bmp"
    DNG	 = ".dng"
    JPG  = ".jpg"
    JPEG = ".jpeg"
    PNG  = ".png"
    PPM  = ".ppm"
    RAF  = ".raf"
    TIF  = ".tif"
    TIFF = ".tiff"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enums."""
        return {
            "arw" : cls.ARW,
            "bmp" : cls.BMP,
            "dng" : cls.DNG,
            "jpg" : cls.JPG,
            "jpeg": cls.JPEG,
            "png" : cls.PNG,
            "ppm" : cls.PPM,
            "raf" : cls.RAF,
            "tif" : cls.TIF,
            "tiff": cls.TIF,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enums."""
        return {
            0: cls.ARW,
            1: cls.BMP,
            2: cls.DNG,
            3: cls.JPG,
            4: cls.JPEG,
            5: cls.PNG,
            6: cls.PPM,
            7: cls.RAF,
            8: cls.TIF,
            9: cls.TIF,
        }
    
    @classmethod
    def from_str(cls, value: str) -> ImageFormat:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ImageFormat:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: ImageFormatType) -> ImageFormat | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ImageFormat):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class MemoryUnit(enum.Enum):
    """Memory units."""
    
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"

    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enums."""
        return {
            "b" : cls.B,
            "kb": cls.KB,
            "mb": cls.MB,
            "gb": cls.GB,
            "tb": cls.TB,
            "pb": cls.PB,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enums."""
        return {
            0: cls.B,
            1: cls.KB,
            2: cls.MB,
            3: cls.GB,
            4: cls.TB,
            5: cls.PB,
        }
    
    @classmethod
    def byte_conversion_mapping(cls) -> dict:
        """Return a dictionary mapping number of bytes to enums."""
        return {
            cls.B : 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MemoryUnit:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value.lower()]
    
    @classmethod
    def from_int(cls, value: int) -> MemoryUnit:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: MemoryUnitType) -> MemoryUnit | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, MemoryUnit):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None
    

class VideoFormat(enum.Enum):
    """Video formats."""
    
    AVI  = ".avi"
    M4V  = ".m4v"
    MKV  = ".mkv"
    MOV  = ".mov"
    MP4  = ".mp4"
    MPEG = ".mpeg"
    MPG  = ".mpg"
    WMV  = ".wmv"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a dictionary mapping strings to enum."""
        return {
            "avi" : cls.AVI,
            "m4v" : cls.M4V,
            "mkv" : cls.MKV,
            "mov" : cls.MOV,
            "mp4" : cls.MP4,
            "mpeg": cls.MPEG,
            "mpg" : cls.MPG,
            "wmv" : cls.WMV,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.AVI,
            1: cls.M4V,
            2: cls.MKV,
            3: cls.MOV,
            4: cls.MP4,
            5: cls.MPEG,
            6: cls.MPG,
            7: cls.WMV,
        }
    
    @classmethod
    def from_str(cls, value: str) -> VideoFormat:
        """Convert a string to an enum."""
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> VideoFormat:
        """Convert an integer to an enum."""
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: VideoFormatType) -> VideoFormat | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, VideoFormat):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None

# endregion
