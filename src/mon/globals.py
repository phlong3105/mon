#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`mon.` package.

Notes:
    - To avoid circular dependency, only define constants of basic/atomic types.
      The same goes for type aliases.
    - The only exception is the enum and factory constants.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "ACCELERATORS",
    "CALLBACKS",
    "CONFIG_FILE_FORMATS",
    "DATAMODULES",
    "DATASETS",
    "DATA_DIR",
    "DETECTORS",
    "DISTANCES",
    "EMBEDDERS",
    "EXTRA_DATASETS",
    "EXTRA_DATASET_STR",
    "EXTRA_METRICS",
    "EXTRA_MODELS",
    "EXTRA_MODEL_STR",
    "FILE_HANDLERS",
    "IMAGE_FILE_FORMATS",
    "LOGGERS",
    "LOSSES",
    "LR_SCHEDULERS",
    "METRICS",
    "MODELS",
    "MON_DIR",
    "MON_EXTRA_DIR",
    "MOTIONS",
    "OBJECTS",
    "OPTIMIZERS",
    "RGB",
    "ROOT_DIR",
    "STRATEGIES",
    "ShapeCode",
    "TORCH_FILE_FORMATS",
    "TRACKERS",
    "VIDEO_FILE_FORMATS",
    "WEIGHTS_FILE_FORMATS",
    "ZOO_DIR",
    # Enum
    "AppleRGB",
    "BBoxFormat",
    "BasicRGB",
    "MemoryUnit",
    "MovingState",
    "Scheme",
    "ShapeCode",
    "Split",
    "Task",
    "TrackState",
]

import os
from typing import Any

from mon.core import dtype as DT, factory, pathlib


# region Directory

current_file = pathlib.Path(__file__).absolute()
ROOT_DIR      = current_file.parents[2]
SRC_DIR       = current_file.parents[1]
MON_DIR       = current_file.parents[0]
MON_EXTRA_DIR = SRC_DIR / "mon_extra"

ZOO_DIR = None
for i, parent in enumerate(current_file.parents):
    if (parent / "zoo").is_dir():
        ZOO_DIR = parent / "zoo"
        break
    if i >= 5:
        break
if ZOO_DIR is None:
    raise Warning(f"Cannot locate the ``zoo`` directory.")

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", None))
DATA_DIR = DATA_DIR or pathlib.Path("/data")
DATA_DIR = DATA_DIR if DATA_DIR.is_dir() else ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    raise Warning(f"Cannot locate the ``data`` directory.")

# endregion


# region Enum

# region Color

class RGB(DT.Enum):
    """138 RGB colors."""
    
    MAROON                  = (128, 0  , 0)
    DARK_RED                = (139, 0  , 0)
    BROWN                   = (165, 42 , 42)
    FIREBRICK               = (178, 34 , 34)
    CRIMSON                 = (220, 20 , 60)
    RED                     = (255, 0  , 0)
    TOMATO                  = (255, 99 , 71)
    CORAL                   = (255, 127, 80)
    INDIAN_RED              = (205, 92 , 92)
    LIGHT_CORAL             = (240, 128, 128)
    DARK_SALMON             = (233, 150, 122)
    SALMON                  = (250, 128, 114)
    LIGHT_SALMON            = (255, 160, 122)
    ORANGE_RED              = (255, 69 , 0)
    DARK_ORANGE             = (255, 140, 0)
    ORANGE                  = (255, 165, 0)
    GOLD                    = (255, 215, 0)
    DARK_GOLDEN_ROD         = (184, 134, 11)
    GOLDEN_ROD              = (218, 165, 32)
    PALE_GOLDEN_ROD         = (238, 232, 170)
    DARK_KHAKI              = (189, 183, 107)
    KHAKI                   = (240, 230, 140)
    OLIVE                   = (128, 128, 0)
    YELLOW                  = (255, 255, 0)
    YELLOW_GREEN            = (154, 205, 50)
    DARK_OLIVE_GREEN        = (85 , 107, 47)
    OLIVE_DRAB              = (107, 142, 35)
    LAWN_GREEN              = (124, 252, 0)
    CHART_REUSE             = (127, 255, 0)
    GREEN_YELLOW            = (173, 255, 47)
    DARK_GREEN              = (0  , 100, 0)
    GREEN                   = (0  , 128, 0)
    FOREST_GREEN            = (34 , 139, 34)
    LIME                    = (0  , 255, 0)
    LIME_GREEN              = (50 , 205, 50)
    LIGHT_GREEN             = (144, 238, 144)
    PALE_GREEN              = (152, 251, 152)
    DARK_SEA_GREEN          = (143, 188, 143)
    MEDIUM_SPRING_GREEN     = (0  , 250, 154)
    SPRING_GREEN            = (0  , 255, 127)
    SEA_GREEN               = (46 , 139, 87)
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_SEA_GREEN        = (60 , 179, 113)
    LIGHT_SEA_GREEN         = (32 , 178, 170)
    DARK_SLATE_GRAY         = (47 , 79 , 79)
    TEAL                    = (0  , 128, 128)
    DARK_CYAN               = (0  , 139, 139)
    AQUA                    = (0  , 255, 255)
    CYAN                    = (0  , 255, 255)
    LIGHT_CYAN              = (224, 255, 255)
    DARK_TURQUOISE          = (0  , 206, 209)
    TURQUOISE               = (64 , 224, 208)
    MEDIUM_TURQUOISE        = (72 , 209, 204)
    PALE_TURQUOISE          = (175, 238, 238)
    AQUA_MARINE             = (127, 255, 212)
    POWDER_BLUE             = (176, 224, 230)
    CADET_BLUE              = (95 , 158, 160)
    STEEL_BLUE              = (70 , 130, 180)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    DEEP_SKY_BLUE           = (0  , 191, 255)
    DODGER_BLUE             = (30 , 144, 255)
    LIGHT_BLUE              = (173, 216, 230)
    SKY_BLUE                = (135, 206, 235)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    MIDNIGHT_BLUE           = (25 , 25 , 112)
    NAVY                    = (0  , 0  , 128)
    DARK_BLUE               = (0  , 0  , 139)
    MEDIUM_BLUE             = (0  , 0  , 205)
    BLUE                    = (0  , 0  , 255)
    ROYAL_BLUE              = (65 , 105, 225)
    BLUE_VIOLET             = (138, 43 , 226)
    INDIGO                  = (75 , 0  , 130)
    DARK_SLATE_BLUE         = (72 , 61 , 139)
    SLATE_BLUE              = (106, 90 , 205)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_PURPLE           = (147, 112, 219)
    DARK_MAGENTA            = (139, 0  , 139)
    DARK_VIOLET             = (148, 0  , 211)
    DARK_ORCHID             = (153, 50 , 204)
    MEDIUM_ORCHID           = (186, 85 , 211)
    PURPLE                  = (128, 0  , 128)
    THISTLE                 = (216, 191, 216)
    PLUM                    = (221, 160, 221)
    VIOLET                  = (238, 130, 238)
    MAGENTA                 = (255, 0  , 255)
    ORCHID                  = (218, 112, 214)
    MEDIUM_VIOLET_RED       = (199, 21 , 133)
    PALE_VIOLET_RED         = (219, 112, 147)
    DEEP_PINK               = (255, 20 , 147)
    HOT_PINK                = (255, 105, 180)
    LIGHT_PINK              = (255, 182, 193)
    PINK                    = (255, 192, 203)
    ANTIQUE_WHITE           = (250, 235, 215)
    BEIGE                   = (245, 245, 220)
    BISQUE                  = (255, 228, 196)
    BLANCHED_ALMOND         = (255, 235, 205)
    WHEAT                   = (245, 222, 179)
    CORN_SILK               = (255, 248, 220)
    LEMON_CHIFFON           = (255, 250, 205)
    LIGHT_GOLDEN_ROD_YELLOW = (250, 250, 210)
    LIGHT_YELLOW            = (255, 255, 224)
    SADDLE_BROWN            = (139, 69 , 19)
    SIENNA                  = (160, 82 , 45)
    CHOCOLATE               = (210, 105, 30)
    PERU                    = (205, 133, 63)
    SANDY_BROWN             = (244, 164, 96)
    BURLY_WOOD              = (222, 184, 135)
    TAN                     = (210, 180, 140)
    ROSY_BROWN              = (188, 143, 143)
    MOCCASIN                = (255, 228, 181)
    NAVAJO_WHITE            = (255, 222, 173)
    PEACH_PUFF              = (255, 218, 185)
    MISTY_ROSE              = (255, 228, 225)
    LAVENDER_BLUSH          = (255, 240, 245)
    LINEN                   = (250, 240, 230)
    OLD_LACE                = (253, 245, 230)
    PAPAYA_WHIP             = (255, 239, 213)
    SEA_SHELL               = (255, 245, 238)
    MINT_CREAM              = (245, 255, 250)
    SLATE_GRAY              = (112, 128, 144)
    LIGHT_SLATE_GRAY        = (119, 136, 153)
    LIGHT_STEEL_BLUE        = (176, 196, 222)
    LAVENDER                = (230, 230, 250)
    FLORAL_WHITE            = (255, 250, 240)
    ALICE_BLUE              = (240, 248, 255)
    GHOST_WHITE             = (248, 248, 255)
    HONEYDEW                = (240, 255, 240)
    IVORY                   = (255, 255, 240)
    AZURE                   = (240, 255, 255)
    SNOW                    = (255, 250, 250)
    BLACK                   = (0  , 0  , 0)
    DIM_GRAY                = (105, 105, 105)
    GRAY                    = (128, 128, 128)
    DARK_GRAY               = (169, 169, 169)
    SILVER                  = (192, 192, 192)
    LIGHT_GRAY              = (211, 211, 211)
    GAINSBORO               = (220, 220, 220)
    WHITE_SMOKE             = (245, 245, 245)
    WHITE                   = (255, 255, 255)


class AppleRGB(DT.Enum):
    """Apple's 12 RGB colors."""
    
    GRAY   = (128, 128, 128)
    RED    = (255, 59 , 48)
    GREEN  = (52 , 199, 89)
    BLUE   = (0  , 122, 255)
    ORANGE = (255, 149, 5)
    YELLOW = (255, 204, 0)
    BROWN  = (162, 132, 94)
    PINK   = (255, 45 , 85)
    PURPLE = (88 , 86 , 214)
    TEAL   = (90 , 200, 250)
    INDIGO = (85 , 190, 240)
    BLACK  = (0  , 0  , 0)
    WHITE  = (255, 255, 255)


class BasicRGB(DT.Enum):
    """12 basic RGB colors."""
    
    BLACK   = (0  , 0  , 0)
    WHITE   = (255, 255, 255)
    RED     = (255, 0  , 0)
    LIME    = (0  , 255, 0)
    BLUE    = (0  , 0  , 255)
    YELLOW  = (255, 255, 0)
    CYAN    = (0  , 255, 255)
    MAGENTA = (255, 0  , 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128, 0  , 0)
    OLIVE   = (128, 128, 0)
    GREEN   = (0  , 128, 0)
    PURPLE  = (128, 0  , 128)
    TEAL    = (0  , 128, 128)
    NAVY    = (0  , 0  , 128)

# endregion


# region Geometry

class BBoxFormat(DT.Enum):
    """Bounding box formats.
    
    CX, CY: refers to a center of bounding box.
    W, H: refers to the width and height of bounding box.
    N: refers to the normalized value in the range :math:`[0.0, 1.0]`:
        x_norm = absolute_x / image_width
        height_norm = absolute_height / image_height
    """
    
    XYXY    = "pascal_voc"
    XYWH    = "coco"
    CXCYWHN = "yolo"
    XYXYN   = "albumentations"
    VOC     = "pascal_voc"
    COCO    = "coco"
    YOLO    = "yolo"
    
    @classmethod
    def str_mapping(cls) -> dict[str, BBoxFormat]:
        """Returns a :class:`dict` mapping :class:`str` to enum."""
        return {
            "xyxy"          : cls.XYXY,
            "xywh"          : cls.XYWH,
            "cxcyn"         : cls.CXCYWHN,
            "albumentations": cls.XYXYN,
            "pascal_voc"    : cls.VOC,
            "coco"          : cls.COCO,
            "yolo"          : cls.YOLO,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, BBoxFormat]:
        """Returns a :class:`dict` mapping :class:`int` to enum."""
        return {
            0: cls.XYXY,
            1: cls.XYWH,
            2: cls.CXCYWHN,
            3: cls.XYXYN,
            4: cls.VOC,
            5: cls.COCO,
            6: cls.YOLO,
        }
    
    @classmethod
    def from_str(cls, value: str) -> BBoxFormat:
        """Converts a :class:`str` to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value.lower()}.")
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BBoxFormat:
        """Convert an :class:`int` to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> BBoxFormat | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, BBoxFormat):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


class ShapeCode(DT.Enum):
    """Shape conversion code."""
    
    # Bounding box
    SAME       = 0
    XYXY2XYWH  = 1
    XYXY2CXCYN = 2
    XYWH2XYXY  = 3
    XYWH2CXCYN = 4
    CXCYN2XYXY = 5
    CXCYN2XYWH = 6
    VOC2COCO   = 7
    VOC2YOLO   = 8
    COCO2VOC   = 9
    COCO2YOLO  = 10
    YOLO2VOC   = 11
    YOLO2COCO  = 12
    
    @classmethod
    def str_mapping(cls) -> dict[str, ShapeCode]:
        """Returns a :class:`dict` mapping :class:`str` to enum."""
        return {
            "same"         : cls.SAME,
            "xyxy_to_xywh" : cls.XYXY2XYWH,
            "xyxy_to_cxcyn": cls.XYXY2CXCYN,
            "xywh_to_xyxy" : cls.XYWH2XYXY,
            "xywh_to_cxcyn": cls.XYWH2CXCYN,
            "cxcyn_to_xyxy": cls.CXCYN2XYXY,
            "cxcyn_to_xywh": cls.CXCYN2XYWH,
            "voc_to_coco"  : cls.VOC2COCO,
            "voc_to_yolo"  : cls.VOC2YOLO,
            "coco_to_voc"  : cls.COCO2VOC,
            "coco_to_yolo" : cls.COCO2YOLO,
            "yolo_to_voc"  : cls.YOLO2VOC,
            "yolo_to_coco" : cls.YOLO2COCO,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, ShapeCode]:
        """Returns a :class:`dict` mapping :class:`int` to enum."""
        return {
            0 : cls.SAME,
            1 : cls.XYXY2XYWH,
            2 : cls.XYXY2CXCYN,
            3 : cls.XYWH2XYXY,
            4 : cls.XYWH2CXCYN,
            5 : cls.CXCYN2XYXY,
            6 : cls.CXCYN2XYWH,
            7 : cls.VOC2COCO,
            8 : cls.VOC2YOLO,
            9 : cls.COCO2VOC,
            10: cls.COCO2YOLO,
            11: cls.YOLO2VOC,
            12: cls.YOLO2COCO,
        }
    
    @classmethod
    def from_str(cls, value: str) -> ShapeCode:
        """Converts a :class:`str` to an enum."""
        if value.lower() not in cls.str_mapping():
            parts = value.split("_to_")
            if parts[0] == parts[1]:
                return cls.SAME
            else:
                raise ValueError(f":param:`value` must be a valid enum key, but got {value.lower()}.")
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ShapeCode:
        """Convert an :class:`int` to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> ShapeCode | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ShapeCode):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None

# endregion


# region Tracking

class TrackState(DT.Enum):
    
    """Enumeration type for a single target track state.
    
    Newly created tracks are classified as `NEW` until enough evidence has been
    collected. Then, the track state is changed to `TRACKED`. Tracks that are no
    longer alive are classified as `REMOVED` to mark them for removal from the
    set of active tracks.
    """
    NEW      = 0
    TRACKED  = 1
    LOST     = 2
    REMOVED  = 3
    REPLACED = 4
    COUNTED  = 5
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a :class:`dict` mapping :class:`str` to enums."""
        return {
            "new"     : cls.NEW,
            "tracked" : cls.TRACKED,
            "lost"    : cls.LOST,
            "removed" : cls.REMOVED,
            "replaced": cls.REPLACED,
            "counted" : cls.COUNTED,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Return a :class:`dict` mapping :class:`int` to enums."""
        return {
            0: cls.NEW,
            1: cls.TRACKED,
            2: cls.LOST,
            3: cls.REMOVED,
            4: cls.REPLACED,
            5: cls.COUNTED,
        }
    
    @classmethod
    def from_str(cls, value: str) -> MovingState:
        """Convert a :class:`str` to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value.lower()}.")
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> MovingState:
        """Convert an :class:`int` to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value}.")
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


class MovingState(DT.Enum):
    
    """The tracking state of an object when moving through the camera."""
    CANDIDATE     = 1  # Preliminary state.
    CONFIRMED     = 2  # Confirmed the Detection is a road_objects eligible for counting.
    COUNTING      = 3  # Object is in the counting zone/counting state.
    TO_BE_COUNTED = 4  # Mark object to be counted somewhere in this loop iteration.
    COUNTED       = 5  # Mark object has been counted.
    EXITING       = 6  # Mark object for exiting the ROI or image frame. Let's it die by itself.
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Return a :class:`dict` mapping :class:`str` to enums."""
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
        """Return a :class:`dict` mapping :class:`int` to enums."""
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
        """Convert a :class:`str` to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value.lower()}.")
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> MovingState:
        """Convert an :class:`int` to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value}.")
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


class MemoryUnit(DT.Enum):
    """Memory units."""
    
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"
    
    @classmethod
    def str_mapping(cls) -> dict[str, MemoryUnit]:
        """Return a :class:`dict` mapping :class:`str` to enums."""
        return {
            "b" : cls.B,
            "kb": cls.KB,
            "mb": cls.MB,
            "gb": cls.GB,
            "tb": cls.TB,
            "pb": cls.PB,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, MemoryUnit]:
        """Return a :class:`dict` mapping :class:`int` to enums."""
        return {
            0: cls.B,
            1: cls.KB,
            2: cls.MB,
            3: cls.GB,
            4: cls.TB,
            5: cls.PB,
        }
    
    @classmethod
    def byte_conversion_mapping(cls) -> dict[MemoryUnit, int]:
        """Return a :class:`dict` mapping number of bytes to enums."""
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
        """Convert a :class:`str` to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value.lower()}.")
        return cls.str_mapping()[value.lower()]
    
    @classmethod
    def from_int(cls, value: int) -> MemoryUnit:
        """Convert an :class:`int` to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(f":param:`value` must be a valid enum key, but got {value}.")
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> MemoryUnit | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, MemoryUnit):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


class Task(DT.Enum):
    """Task types."""
    
    CLASSIFY = "classify"
    DEBLUR   = "deblur"
    DEHAZE   = "dehaze"
    DENOISE  = "denoise"
    DEPTH    = "depth"
    DERAIN   = "derain"
    DESNOW   = "desnow"
    DETECT   = "detect"
    INPAINT  = "inpaint"
    LES      = "les"
    LLIE     = "llie"
    POSE     = "pose"
    SEGMENT  = "segment"
    TRACK    = "track"


class RunMode(DT.Enum):
    """Run modes."""
    
    TRAIN   = "train"
    PREDICT = "predict"
    METRIC  = "metric"
    PLOT    = "plot"
    

class Scheme(DT.Enum):
    """Learning schemes."""
    
    TRADITIONAL  = "traditional"   # traditional method (no learning)
    SUPERVISED   = "supervised"    # supervised learning
    UNSUPERVISED = "unsupervised"  # unsupervised learning
    ONLINE       = "online"        # online learning
    INSTANCE     = "instance"      # one-instance learning
    FEWSHOT      = "few-shot"      # few-shot learning
    ONESHOT      = "one-shot"      # one-shot learning
    ZEROSHOT     = "zero-shot"     # zero-shot learning
    

class Split(DT.Enum):
    """Dataset split types."""
    
    TRAIN   = "train"
    VAL     = "val"
    TEST    = "test"
    PREDICT = "predict"

# endregion


# region Constants

CONFIG_FILE_FORMATS  = [".config", ".cfg", ".yaml", ".yml", ".py", ".json", ".names", ".txt"]
IMAGE_FILE_FORMATS   = [".arw", ".bmp", ".dng", ".jpg", ".jpeg", ".png", ".ppm", ".raf", ".tif", ".tiff"]
VIDEO_FILE_FORMATS   = [".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"]
TORCH_FILE_FORMATS   = [".pt", ".pth", ".weights", ".ckpt", ".tar", ".onnx"]
WEIGHTS_FILE_FORMATS = [".pt", ".pth", ".onnx"]

# List 3rd party modules
EXTRA_DATASET_STR = "[extra]"
EXTRA_MODEL_STR   = "[extra]"
EXTRA_DATASETS    = {
    # region detect
        "aicity_2024_fisheye8k": {
        "tasks"         : [Task.DETECT],
        "splits"        : [Split.TRAIN, Split.VAL, Split.TEST],
        "has_test_label": False,
    },
    # endregion
}
EXTRA_METRICS     = {
    # FR
    "ahiq"               : {"metric_mode": "FR", "lower_is_better": False, },
    "ckdn"               : {"metric_mode": "FR", "lower_is_better": False, },
    "cw-ssim"            : {"metric_mode": "FR", "lower_is_better": False, },
    "cw_ssim"            : {"metric_mode": "FR", "lower_is_better": False, },
    "dists"              : {"metric_mode": "FR", "lower_is_better": True , },
    "fsim"               : {"metric_mode": "FR", "lower_is_better": False, },
    "gmsd"               : {"metric_mode": "FR", "lower_is_better": True , },
    "haarpsi"            : {"metric_mode": "FR", "lower_is_better": False, },
    "lpips"              : {"metric_mode": "FR", "lower_is_better": True , },
    "lpips-vgg"          : {"metric_mode": "FR", "lower_is_better": True , },
    "lpips_vgg"          : {"metric_mode": "FR", "lower_is_better": True , },
    "mad"                : {"metric_mode": "FR", "lower_is_better": False, },
    "mdsi"               : {"metric_mode": "FR", "lower_is_better": False, },
    "ms-gmsd"            : {"metric_mode": "FR", "lower_is_better": False, },
    "ms-ssim"            : {"metric_mode": "FR", "lower_is_better": False, },
    "ms_gmsd"            : {"metric_mode": "FR", "lower_is_better": False, },
    "ms_ssim"            : {"metric_mode": "FR", "lower_is_better": False, },
    "nlpd"               : {"metric_mode": "FR", "lower_is_better": True , },
    "psnr"               : {"metric_mode": "FR", "lower_is_better": False, },
    "psnry"              : {"metric_mode": "FR", "lower_is_better": False, },
    "ssim"               : {"metric_mode": "FR", "lower_is_better": False, },
    "ssimc"              : {"metric_mode": "FR", "lower_is_better": False, },
    "stlpips"            : {"metric_mode": "FR", "lower_is_better": True , },
    "stlpips-vgg"        : {"metric_mode": "FR", "lower_is_better": True , },
    "stlpips_vgg"        : {"metric_mode": "FR", "lower_is_better": True , },
    "vif"                : {"metric_mode": "FR", "lower_is_better": False, },
    "vsi"                : {"metric_mode": "FR", "lower_is_better": False, },
    # NR
    "brisque"            : {"metric_mode": "NR", "lower_is_better": True , },
    "clipiqa"            : {"metric_mode": "NR", "lower_is_better": False, },
    "clipiqa+"           : {"metric_mode": "NR", "lower_is_better": False, },
    "clipiqa+_rn50_512"  : {"metric_mode": "NR", "lower_is_better": False, },
    "clipiqa+_vitL14_512": {"metric_mode": "NR", "lower_is_better": False, },
    "clipscore"          : {"metric_mode": "NR", "lower_is_better": False, },
    "cnniqa"             : {"metric_mode": "NR", "lower_is_better": False, },
    "dbcnn"              : {"metric_mode": "NR", "lower_is_better": False, },
    "entropy"            : {"metric_mode": "NR", "lower_is_better": False, },
    "fid"                : {"metric_mode": "NR", "lower_is_better": True , },
    "hyperiqa"           : {"metric_mode": "NR", "lower_is_better": False, },
    "laion_aes"          : {"metric_mode": "NR", "lower_is_better": False, },
    "maniqa"             : {"metric_mode": "NR", "lower_is_better": False, },
    "maniqa-kadid"       : {"metric_mode": "NR", "lower_is_better": False, },
    "maniqa-koniq"       : {"metric_mode": "NR", "lower_is_better": False, },
    "maniqa-pipal"       : {"metric_mode": "NR", "lower_is_better": False, },
    "musiq"              : {"metric_mode": "NR", "lower_is_better": False, },
    "musiq-ava"          : {"metric_mode": "NR", "lower_is_better": False, },
    "musiq-koniq"        : {"metric_mode": "NR", "lower_is_better": False, },
    "musiq-paq2piq"      : {"metric_mode": "NR", "lower_is_better": False, },
    "musiq-spaq"         : {"metric_mode": "NR", "lower_is_better": False, },
    "nima"               : {"metric_mode": "NR", "lower_is_better": False, },
    "nima-vgg16-ava"     : {"metric_mode": "NR", "lower_is_better": False, },
    "niqe"               : {"metric_mode": "NR", "lower_is_better": True , },
    "nrqm"               : {"metric_mode": "NR", "lower_is_better": False, },
    "paq2piq"            : {"metric_mode": "NR", "lower_is_better": False, },
    "pi"                 : {"metric_mode": "NR", "lower_is_better": True , },
    "pieapp"             : {"metric_mode": "NR", "lower_is_better": True , },
    "topiq_fr"           : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_fr-pipal"     : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_iaa"          : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_iaa_res50"    : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_nr"           : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_nr-face"      : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_nr-flive"     : {"metric_mode": "NR", "lower_is_better": False, },
    "topiq_nr-spaq"      : {"metric_mode": "NR", "lower_is_better": False, },
    "tres"               : {"metric_mode": "NR", "lower_is_better": False, },
    "tres-flive"         : {"metric_mode": "NR", "lower_is_better": False, },
    "tres-koniq"         : {"metric_mode": "NR", "lower_is_better": False, },
    "tv"                 : {"metric_mode": "NR", "lower_is_better": False, },
    "uranker"            : {"metric_mode": "NR", "lower_is_better": False, },
}
EXTRA_MODELS      = {  # architecture/model (+ variant)
    # region depth
    "depth_anything_v2": {
        "depth_anything_v2_vitb": {
            "tasks"    : [Task.DEPTH],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "depth" / "depth_anything_v2",
            "torch_distributed_launch": True,
        },
        "depth_anything_v2_vits": {
            "tasks"    : [Task.DEPTH],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "depth" / "depth_anything_v2",
            "torch_distributed_launch": True,
        },
        "depth_anything_v2_vitl": {
            "tasks"    : [Task.DEPTH],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "depth" / "depth_anything_v2",
            "torch_distributed_launch": True,
        },
        "depth_anything_v2_vitg": {
            "tasks"    : [Task.DEPTH],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "depth" / "depth_anything_v2",
            "torch_distributed_launch": True,
        },
    },
    # endregion
    # region detect
    "yolor"        : {
        "yolor_d6": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolor",
            "torch_distributed_launch": True,
        },
        "yolor_e6": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolor",
            "torch_distributed_launch": True,
        },
        "yolor_p6": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolor",
            "torch_distributed_launch": True,
        },
        "yolor_w6": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolor",
            "torch_distributed_launch": True,
        },
    },
    "yolov7"       : {
        "yolov7"    : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
        "yolov7_d6" : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
        "yolov7_e6" : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
        "yolov7_e6e": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
        "yolov7_w6" : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
        "yolov7x"   : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov7",
            "torch_distributed_launch": True,
        },
    },
    "yolov8"       : {
        "yolov8n": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
            "torch_distributed_launch": False,
        },
        "yolov8s": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
            "torch_distributed_launch": False,
        },
        "yolov8m": {
            "tasks"    : [Task.DETECT],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
            "torch_distributed_launch": False,
        },
        "yolov8l": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
            "torch_distributed_launch": False,
        },
        "yolov8x": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "ultralytics",
            "torch_distributed_launch": False,
        },
    },
    "yolov9"       : {
        "gelan_c" : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov9",
            "torch_distributed_launch": True,
        },
        "gelan_e" : {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov9",
            "torch_distributed_launch": True,
        },
        "yolov9_c": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov9",
            "torch_distributed_launch": True,
        },
        "yolov9_e": {
            "tasks"    : [Task.DETECT],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "detect" / "yolov9",
            "torch_distributed_launch": True,
        },
    },
    # endregion
    # region enhance/dehaze
    "zid"          : {
        "zid": {
            "tasks"    : [Task.DEHAZE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "dehaze" / "zid",
            "torch_distributed_launch": False,
        },
    },
    # endregion
    # region enhance/llie
    "colie"        : {
        "colie": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "colie",
            "torch_distributed_launch": False,
        },
    },
    "dccnet"       : {
        "dccnet": {
                "tasks"    : [Task.LLIE],
                "schemes"  : [Scheme.SUPERVISED],
                "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "dccnet",
                "torch_distributed_launch": True,
            },
    },
    "enlightengan" : {
        "enlightengan": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.UNSUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "enlightengan",
            "torch_distributed_launch": True,
        },
    },
    "gsad"         : {
        "gsad": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "gsad",
            "torch_distributed_launch": True,
        },
    },
    "hvi_cidnet"   : {
        "hvi_cidnet": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "hvi_cidnet",
            "torch_distributed_launch": True,
        },
    },
    "kind"         : {
        "kind": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "kind",
            "torch_distributed_launch": True,
        },
    },
    "lime"         : {
        "lime": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.TRADITIONAL],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "lime",
            "torch_distributed_launch": True,
        },
    },
    "llflow"       : {
        "llflow": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "llflow",
            "torch_distributed_launch": True,
        },
    },
    "llunet++"     : {
        "llunet++": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "llunet++",
            "torch_distributed_launch": True,
        },
    },
    "mtfe"         : {
        "mtfe": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "mtfe",
            "torch_distributed_launch": True,
        },
    },
    "pie"          : {
        "pie": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.TRADITIONAL],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "pie",
            "torch_distributed_launch": True,
        },
    },
    "psenet"       : {
        "psenet": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "psenet",
            "torch_distributed_launch": False,
        }
    },
    "quadprior"    : {
        "quadprior": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "quadprior",
            "torch_distributed_launch": False,
        }
    },
    "retinexformer": {
        "retinexformer": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "retinexformer",
            "torch_distributed_launch": True,
        },
    },
    "retinexnet"   : {
        "retinexnet": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "retinexnet",
            "torch_distributed_launch": True,
        },
    },
    "ruas"         : {
        "ruas": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "ruas",
            "torch_distributed_launch": True,
        },
    },
    "sci"          : {
        "sci": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "sci",
            "torch_distributed_launch": True,
        },
    },
    "sgz"          : {
        "sgz": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "sgz",
            "torch_distributed_launch": True,
        },
    },
    "snr"          : {
        "snr": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "snr",
            "torch_distributed_launch": True,
        },
    },
    "stablellve"   : {
        "stablellve": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "stablellve",
            "torch_distributed_launch": True,
        },
    },
    "uretinexnet"  : {
        "uretinexnet": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "uretinexnet",
            "torch_distributed_launch": True,
        },
    },
    "utvnet"       : {
        "utvnet": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "utvnet",
            "torch_distributed_launch": True,
        },
    },
    "zero_dce"     : {
        "zero_dce": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "zero_dce",
            "torch_distributed_launch": True,
        },
    },
    "zero_dce++"   : {
        "zero_dce++": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "zero_dce++",
            "torch_distributed_launch": True,
        },
    },
    "zero_didce"   : {
        "zero_didce": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "zero_didce",
            "torch_distributed_launch": True,
        },
    },
    "zeroig"       : {
        "zeroig": {
            "tasks"    : [Task.LLIE],
            "schemes"  : [Scheme.ZEROSHOT],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "llie" / "zeroig",
            "torch_distributed_launch": True,
        },
    },
    # endregion
    # region enhance/multitask
    "restormer"    : {
        "restormer": {
            "tasks"    : [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW, Task.LLIE],
            "schemes"  : [Scheme.SUPERVISED],
            "model_dir": MON_EXTRA_DIR / "vision" / "enhance" / "multitask" / "restormer",
            "torch_distributed_launch": True,
        },
    },
    # endregion
}

# endregion


# region Factory

ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
DETECTORS     = factory.Factory(name="Detectors")
DISTANCES     = factory.Factory(name="Distances")
EMBEDDERS     = factory.Factory(name="Embedders")
FILE_HANDLERS = factory.Factory(name="FileHandlers")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.ModelFactory(name="Models")
MOTIONS       = factory.Factory(name="Motions")
OBJECTS       = factory.Factory(name="Objects")
OPTIMIZERS    = factory.Factory(name="Optimizers")
STRATEGIES    = factory.Factory(name="Strategies")
TRACKERS      = factory.Factory(name="Trackers")

# endregion
