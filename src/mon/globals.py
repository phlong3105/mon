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
    "ACCELERATORS", "AppleRGB", "BBoxFormat", "BIN_DIR", "BasicRGB",
    "BorderType", "CALLBACKS", "DATAMODULES", "DATASETS", "DATA_DIR",
    "DETECTORS", "DISTANCES", "DOCS_DIR", "FILE_HANDLERS", "IMG_MEAN",
    "IMG_STD", "ImageFormat", "InterpolationMode", "LAYERS", "LOGGERS",
    "LOSSES", "LR_SCHEDULERS", "METRICS", "MODELS", "MOTIONS", "MemoryUnit",
    "ModelPhase", "MovingState", "OBJECTS", "OPTIMIZERS", "PaddingMode", "RGB",
    "ROOT_DIR", "RUN_DIR", "Reduction", "SOURCE_DIR", "STRATEGIES", "ShapeCode",
    "TEST_DIR", "TRACKERS", "TRANSFORMS", "VideoFormat", "ZOO_DIR",
]

import os
from typing import Any

import cv2

from mon.foundation import enum, factory, pathlib

# region Directory

_current_file = pathlib.Path(__file__).absolute()
PACKAGE_DIR   = _current_file.parents[0]
SOURCE_DIR    = _current_file.parents[1]
ROOT_DIR      = _current_file.parents[2]
BIN_DIR       = ROOT_DIR / "bin"
DOCS_DIR      = ROOT_DIR / "docs"
RUN_DIR       = ROOT_DIR / "run"
TEST_DIR      = ROOT_DIR / "test"

ZOO_DIR = PACKAGE_DIR / "zoo"
if not ZOO_DIR.is_dir():
    ZOO_DIR = SOURCE_DIR / "zoo"
if not ZOO_DIR.is_dir():
    ZOO_DIR = ROOT_DIR / "zoo"

DATA_DIR = os.getenv("DATA_DIR", None)
if DATA_DIR is None:
    DATA_DIR = pathlib.Path("/data")
else:
    DATA_DIR = pathlib.Path(DATA_DIR)
if not DATA_DIR.is_dir():
    DATA_DIR = ROOT_DIR / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = ""

# endregion


# region Factory

ACCELERATORS  = factory.Factory(name="Accelerators")
CALLBACKS     = factory.Factory(name="Callbacks")
DATAMODULES   = factory.Factory(name="DataModules")
DATASETS      = factory.Factory(name="Datasets")
DETECTORS     = factory.Factory(name="Detectors")
DISTANCES     = factory.Factory(name="Distances")
FILE_HANDLERS = factory.Factory(name="FileHandlers")
LAYERS        = factory.Factory(name="Layers")
LOGGERS       = factory.Factory(name="Loggers")
LOSSES        = factory.Factory(name="Losses")
LR_SCHEDULERS = factory.Factory(name="LRSchedulers")
METRICS       = factory.Factory(name="Metrics")
MODELS        = factory.Factory(name="Models")
MOTIONS       = factory.Factory(name="Motions")
OBJECTS       = factory.Factory(name="Objects")
OPTIMIZERS    = factory.Factory(name="Optimizers")
STRATEGIES    = factory.Factory(name="Strategies")
TRACKERS      = factory.Factory(name="Trackers")
TRANSFORMS    = factory.Factory(name="Transforms")

# endregion


# region Enum

# Color

class RGB(enum.Enum):
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


class AppleRGB(enum.Enum):
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


class BasicRGB(enum.Enum):
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


# Model

class ModelPhase(enum.Enum):
    """Model training phases."""
    
    TRAINING = "training"
    # Produce predictions, calculate losses and metrics, update weights at
    # the end of each epoch/step.
    TESTING = "testing"
    # Produce predictions, calculate losses and metrics, DO NOT update weights
    # at the end of each epoch/step.
    INFERENCE = "inference"
    
    # Produce predictions ONLY.
    
    @classmethod
    def str_mapping(cls) -> dict[str, ModelPhase]:
        """Return a dictionary mapping strings to enum."""
        return {
            "training" : cls.TRAINING,
            "testing"  : cls.TESTING,
            "inference": cls.INFERENCE,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, ModelPhase]:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.TRAINING,
            1: cls.TESTING,
            2: cls.INFERENCE,
        }
    
    @classmethod
    def from_str(cls, value: str) -> ModelPhase:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ModelPhase:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> ModelPhase | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ModelPhase):
            return value
        elif isinstance(value, str):
            return cls.from_str(value)
        elif isinstance(value, int):
            return cls.from_int(value)
        return None


class Reduction(enum.Enum):
    """Tensor reduction options"""
    
    NONE         = "none"
    MEAN         = "mean"
    SUM          = "sum"
    WEIGHTED_SUM = "weighted_sum"
    
    @classmethod
    def str_mapping(cls) -> dict[str, Reduction]:
        """Return a dictionary mapping strings to enum."""
        return {
            "none"        : cls.NONE,
            "mean"        : cls.MEAN,
            "sum"         : cls.SUM,
            "weighted_sum": cls.WEIGHTED_SUM
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, Reduction]:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.NONE,
            1: cls.MEAN,
            2: cls.SUM,
            3: cls.WEIGHTED_SUM,
        }
    
    @classmethod
    def from_str(cls, value: str) -> Reduction:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> Reduction:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> Reduction | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, Reduction):
            return value
        elif isinstance(value, str):
            return cls.from_str(value)
        elif isinstance(value, int):
            return cls.from_int(value)
        return None


# Geometry

class BBoxFormat(enum.Enum):
    """Bounding box formats.
    
    CX, CY: refers to a center of bounding box.
    W, H: refers to width and height of bounding box.
    N: refers to the normalized value in the range [0.0, 1.0]:
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
        """Returns a dictionary mapping strings to enum."""
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
        """Returns a dictionary mapping integers to enum."""
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
        """Converts a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BBoxFormat:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
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


class ShapeCode(enum.Enum):
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
        """Returns a dictionary mapping strings to enum."""
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
        """Returns a dictionary mapping integers to enum."""
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
        """Converts a string to an enum."""
        if value.lower() not in cls.str_mapping():
            parts = value.split("_to_")
            if parts[0] == parts[1]:
                return cls.SAME
            else:
                raise ValueError(
                    f"value must be a valid enum key, but got {value.lower()}."
                )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ShapeCode:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
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
    

# Image

class BorderType(enum.Enum):
    
    CONSTANT  = "constant"
    CIRCULAR  = "circular"
    REFLECT   = "reflect"
    REPLICATE = "replicate"
    
    @classmethod
    def str_mapping(cls) -> dict[str, BorderType]:
        """Return a dictionary mapping strings to enum."""
        return {
            "constant" : cls.CONSTANT,
            "circular" : cls.CIRCULAR,
            "reflect"  : cls.REFLECT,
            "replicate": cls.REPLICATE,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, BorderType]:
        """Return a dictionary mapping integers to enum."""
        return {
            0: cls.CONSTANT,
            1: cls.CIRCULAR,
            2: cls.REFLECT,
            3: cls.REPLICATE,
        }
    
    @classmethod
    def from_str(cls, value: str) -> BorderType:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BorderType:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> BorderType | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, BorderType):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


class InterpolationMode(enum.Enum):
    
    AREA          = "area"
    BICUBIC       = "bicubic"
    BILINEAR      = "bilinear"
    CUBIC         = "cubic"
    LANCZOS4      = "lanczos4"
    LINEAR        = "linear"
    LINEAR_EXACT  = "linear_exact"
    MAX           = "max"
    NEAREST       = "nearest"
    NEAREST_EXACT = "nearest_exact"
    
    @classmethod
    def str_mapping(cls) -> dict[str, InterpolationMode]:
        """Return a dictionary mapping strings to enum."""
        return {
            "area"         : cls.AREA,
            "bicubic"      : cls.BICUBIC,
            "bilinear"     : cls.BILINEAR,
            "cubic"        : cls.CUBIC,
            "lanczos4"     : cls.LANCZOS4,
            "linear"       : cls.LINEAR,
            "linear_exact" : cls.LINEAR_EXACT,
            "max"          : cls.MAX,
            "nearest"      : cls.NEAREST,
            "nearest_exact": cls.NEAREST_EXACT,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, InterpolationMode]:
        """Return a dictionary mapping integers to enum."""
        return {
            0 : cls.AREA,
            1 : cls.BICUBIC,
            2 : cls.BILINEAR,
            6 : cls.CUBIC,
            7 : cls.LANCZOS4,
            8 : cls.LINEAR,
            9 : cls.LINEAR_EXACT,
            10: cls.MAX,
            11: cls.NEAREST,
            12: cls.NEAREST_EXACT,
        }
    
    @classmethod
    def cv_modes_mapping(cls) -> dict[InterpolationMode, int]:
        """Returns a dictionary mapping cv2 interpolation mode to enum."""
        return {
            cls.AREA         : cv2.INTER_AREA,
            cls.BICUBIC      : cv2.INTER_AREA,
            cls.BILINEAR     : cv2.INTER_LINEAR,
            cls.CUBIC        : cv2.INTER_CUBIC,
            cls.LANCZOS4     : cv2.INTER_LANCZOS4,
            cls.LINEAR       : cv2.INTER_LINEAR,
            cls.LINEAR_EXACT : cv2.INTER_LINEAR_EXACT,
            cls.MAX          : cv2.INTER_MAX,
            cls.NEAREST      : cv2.INTER_NEAREST,
            cls.NEAREST_EXACT: cv2.INTER_NEAREST_EXACT
        }
    
    @classmethod
    def from_str(cls, value: str) -> InterpolationMode:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> InterpolationMode:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> InterpolationMode | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, InterpolationMode):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


class PaddingMode(enum.Enum):
    
    CONSTANT    = "constant"
    # For torch compatibility
    CIRCULAR    = "circular"
    REFLECT     = "reflect"
    REPLICATE   = "replicate"
    # For numpy compatibility
    EDGE        = "edge"
    EMPTY       = "empty"
    LINEAR_RAMP = "linear_ramp"
    MAXIMUM     = "maximum"
    MEAN        = "mean"
    MEDIAN      = "median"
    MINIMUM     = "minimum"
    SYMMETRIC   = "symmetric"
    WRAP        = "wrap"
    
    @classmethod
    def str_mapping(cls) -> dict[str, PaddingMode]:
        """Return a dictionary mapping strings to enum."""
        return {
            "constant"   : cls.CONSTANT,
            "circular"   : cls.CIRCULAR,
            "reflect"    : cls.REFLECT,
            "replicate"  : cls.REPLICATE,
            "edge"       : cls.EDGE,
            "empty"      : cls.EMPTY,
            "linear_ramp": cls.LINEAR_RAMP,
            "maximum"    : cls.MAXIMUM,
            "mean"       : cls.MEAN,
            "median"     : cls.MEDIAN,
            "minimum"    : cls.MINIMUM,
            "symmetric"  : cls.SYMMETRIC,
            "wrap"       : cls.WRAP,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, PaddingMode]:
        """Return a dictionary mapping integers to enum."""
        return {
            0 : cls.CONSTANT,
            1 : cls.CIRCULAR,
            2 : cls.REFLECT,
            3 : cls.REPLICATE,
            4 : cls.EDGE,
            5 : cls.EMPTY,
            6 : cls.LINEAR_RAMP,
            7 : cls.MAXIMUM,
            8 : cls.MEAN,
            9 : cls.MEDIAN,
            10: cls.MINIMUM,
            11: cls.SYMMETRIC,
            12: cls.WRAP,
        }
    
    @classmethod
    def from_str(cls, value: str) -> PaddingMode:
        """Convert a string to an enum."""
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> PaddingMode:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> PaddingMode | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, PaddingMode):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


class ImageFormat(enum.Enum):
    """Image file formats."""
    
    ARW  = ".arw"
    BMP  = ".bmp"
    DNG  = ".dng"
    JPG  = ".jpg"
    JPEG = ".jpeg"
    PNG  = ".png"
    PPM  = ".ppm"
    RAF  = ".raf"
    TIF  = ".tif"
    TIFF = ".tiff"
    
    @classmethod
    def str_mapping(cls) -> dict[str, ImageFormat]:
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
    def int_mapping(cls) -> dict[int, ImageFormat]:
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
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ImageFormat:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value) -> ImageFormat | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, ImageFormat):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
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
    def str_mapping(cls) -> dict[str, VideoFormat]:
        """Return a dictionary mapping strings to enum."""
        return {
            ".avi" : cls.AVI,
            ".m4v" : cls.M4V,
            ".mkv" : cls.MKV,
            ".mov" : cls.MOV,
            ".mp4" : cls.MP4,
            ".mpeg": cls.MPEG,
            ".mpg" : cls.MPG,
            ".wmv" : cls.WMV,
        }
    
    @classmethod
    def int_mapping(cls) -> dict[int, VideoFormat]:
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
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> VideoFormat:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: Any) -> VideoFormat | None:
        """Convert an arbitrary value to an enum."""
        if isinstance(value, VideoFormat):
            return value
        elif isinstance(value, str):
            return cls.from_str(value=value)
        elif isinstance(value, int):
            return cls.from_int(value=value)
        return None


# Misc

class MemoryUnit(enum.Enum):
    """Memory units."""
    
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"
    
    @classmethod
    def str_mapping(cls) -> dict[str, MemoryUnit]:
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
    def int_mapping(cls) -> dict[int, MemoryUnit]:
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
    def byte_conversion_mapping(cls) -> dict[MemoryUnit, int]:
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
        if value.lower() not in cls.str_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value.lower()}."
            )
        return cls.str_mapping()[value.lower()]
    
    @classmethod
    def from_int(cls, value: int) -> MemoryUnit:
        """Convert an integer to an enum."""
        if value not in cls.int_mapping():
            raise ValueError(
                f"value must be a valid enum key, but got {value}."
            )
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


# Tracking

class MovingState(enum.Enum):
    
    """The tracking state of an object when moving through the camera."""
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


# region Value

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# endregion
