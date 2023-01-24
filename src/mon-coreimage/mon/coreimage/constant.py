#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all global constants used across :mod:`mon.coreimage`
package.
"""

from __future__ import annotations

__all__ = [
    "AppleRGB", "BBoxFormat", "BasicRGB", "BorderType", "CFA", "DistanceMetric",
    "IMG_MEAN", "IMG_STD", "InterpolationMode", "PaddingMode", "RGB",
    "VISION_BACKEND", "VisionBackend",
    # Extend :mod:`mon.core.constant`
    "CONTENT_ROOT_DIR", "DATA_DIR", "DOCS_DIR", "FILE_HANDLER", "ImageFormat",
    "MemoryUnit", "PRETRAINED_DIR", "PROJECTS_DIR", "RUNS_DIR",
    "SOURCE_ROOT_DIR", "SNIPPET_DIR", "SRC_DIR", "VideoFormat",
]

from typing import TYPE_CHECKING

import cv2

from mon import core
from mon.core.constant import *

if TYPE_CHECKING:
    from mon.coreimage.typing import (
        BBoxFormatType, BorderTypeType, DistanceMetricType,
        InterpolationModeType, PaddingModeType, VisionBackendType,
    )


# DEFAULT_CROP_PCT = 0.875
IMG_MEAN         = [0.485, 0.456, 0.406]
IMG_STD          = [0.229, 0.224, 0.225]


# region Enum

class AppleRGB(core.Enum):
    """Apple's 12 RGB colors."""
    
    GRAY   = (128, 128, 128)
    RED    = (255,  59,  48)
    GREEN  = ( 52, 199,  89)
    BLUE   = (  0, 122, 255)
    ORANGE = (255, 149,   5)
    YELLOW = (255, 204,   0)
    BROWN  = (162, 132,  94)
    PINK   = (255,  45,  85)
    PURPLE = ( 88,  86, 214)
    TEAL   = ( 90, 200, 250)
    INDIGO = ( 85, 190, 240)
    BLACK  = (  0,   0,   0)
    WHITE  = (255, 255, 255)


class BasicRGB(core.Enum):
    """12 basic RGB colors."""
    
    BLACK   = (  0,   0,   0)
    WHITE   = (255, 255, 255)
    RED     = (255,   0,   0)
    LIME    = (  0, 255,   0)
    BLUE    = (  0,   0, 255)
    YELLOW  = (255, 255,   0)
    CYAN    = (  0, 255, 255)
    MAGENTA = (255,   0, 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    MAROON  = (128,   0,   0)
    OLIVE   = (128, 128,   0)
    GREEN   = (  0, 128,   0)
    PURPLE  = (128,   0, 128)
    TEAL    = (  0, 128, 128)
    NAVY    = (  0,   0, 128)
    

class BBoxFormat(core.Enum):
    """Bounding box formats."""
    
    CXCYAR      = "cxcyar"
    CXCYRH      = "cxcyrh"
    CXCYWH      = "cxcywh"
    CXCYWH_NORM = "cxcywh_norm"
    XYXY        = "xyxy"
    XYWH        = "xywh"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
        return {
            "cxcyar"     : cls.CXCYAR,
            "cxcyrh"     : cls.CXCYRH,
            "cxcywh"     : cls.CXCYWH,
            "cxcywh_norm": cls.CXCYWH_NORM,
            "xyxy"       : cls.XYXY,
            "xywh"       : cls.XYWH,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
        return {
            0: cls.CXCYAR,
            1: cls.CXCYRH,
            2: cls.CXCYWH,
            3: cls.CXCYWH_NORM,
            4: cls.XYXY,
            5: cls.XYWH,
        }
    
    @classmethod
    def from_str(cls, value: str) -> BBoxFormat:
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BBoxFormat:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: BBoxFormatType) -> BBoxFormat | None:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """
        if isinstance(value, BBoxFormat):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class BorderType(core.Enum):
    
    CONSTANT      = "constant"
    CIRCULAR      = "circular"
    REFLECT       = "reflect"
    REPLICATE     = "replicate"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
        return {
            "constant"   : cls.CONSTANT,
            "circular"   : cls.CIRCULAR,
            "reflect"    : cls.REFLECT,
            "replicate"  : cls.REPLICATE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
        return {
            0 : cls.CONSTANT,
            1 : cls.CIRCULAR,
            2 : cls.REFLECT,
            3 : cls.REPLICATE,
        }

    @classmethod
    def from_str(cls, value: str) -> BorderType:
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> BorderType:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: BorderTypeType) -> BorderType | None:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """
        if isinstance(value, BorderType):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class CFA(core.Enum):
    """Define the configuration of the color filter array.

    So far only bayer images is supported and the enum sets the pixel order for
    bayer. Note that this can change due to things like rotations and cropping
    of images. Take care if including the translations in pipeline. This
    implementation is optimized to be reasonably fast, look better than simple
    nearest neighbour. On top of this care is taken to make it reversible going
    raw -> rgb -> raw. the raw samples remain intact during conversion and only
    unknown samples are interpolated.

    Names are based on the OpenCV convention where the BG indicates pixel 1,1
    (counting from 0,0) is blue and its neighbour to the right is green. In that
    case the top left pixel is red. Other options are GB, RG and GR

    Reference:
        https://en.wikipedia.org/wiki/Color_filter_array
    """
    
    BG = 0
    GB = 1
    RG = 2
    GR = 3


class DistanceMetric(core.Enum):
    
    BRAYCURTIS         = "braycurtis"
    CANBERRA           = "canberra"
    CHEBYSHEV          = "chebyshev"
    CITYBLOCK          = "cityblock"
    CORRELATION        = "correlation"
    COSINE             = "cosine"
    DICE               = "dice"
    DIRECTED_HAUSDORFF = "directed_hausdorff"
    EUCLIDEAN          = "euclidean"
    HAMMING            = "hamming"
    JACCARD            = "jaccard"
    JENSENSHANNON      = "jensenshannon"
    KULCZYNSKI1        = "kulczynski1"
    KULSINSKI          = "kulsinski"
    MAHALANOBIS        = "mahalanobis"
    MINKOWSKI          = "minkowski"
    ROGERSTANIMOTO     = "rogerstanimoto"
    RUSSELLRAO         = "russellrao"
    SEUCLIDEAN         = "seuclidean"
    SOKALMICHENER      = "sokalmichener"
    SOKALSNEATH        = "sokalsneath"
    SQEUCLIDEAN        = "sqeuclidean"
    YULE               = "yule"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
        return {
            "braycurtis"        : cls.BRAYCURTIS,
            "canberra"          : cls.CANBERRA,
            "chebyshev"         : cls.CHEBYSHEV,
            "cityblock"         : cls.CITYBLOCK,
            "correlation"       : cls.CORRELATION,
            "cosine"            : cls.COSINE,
            "dice"              : cls.DICE,
            "directed_hausdorff": cls.DIRECTED_HAUSDORFF,
            "euclidean"         : cls.EUCLIDEAN,
            "hamming"           : cls.HAMMING,
            "jaccard"           : cls.JACCARD,
            "jensenshannon"     : cls.JENSENSHANNON,
            "kulczynski1"       : cls.KULCZYNSKI1,
            "kulsinski"         : cls.KULSINSKI,
            "mahalanobis"       : cls.MAHALANOBIS,
            "minkowski"         : cls.MINKOWSKI,
            "rogerstanimoto"    : cls.ROGERSTANIMOTO,
            "russellrao"        : cls.RUSSELLRAO,
            "seuclidean"        : cls.SEUCLIDEAN,
            "sokalmichener"     : cls.SOKALMICHENER,
            "sokalsneath"       : cls.SOKALSNEATH,
            "sqeuclidean"       : cls.SQEUCLIDEAN,
            "yule"              : cls.YULE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
        return {
            0 : cls.BRAYCURTIS,
            1 : cls.CANBERRA,
            2 : cls.CHEBYSHEV,
            3 : cls.CITYBLOCK,
            4 : cls.CORRELATION,
            5 : cls.COSINE,
            6 : cls.DICE,
            7 : cls.DIRECTED_HAUSDORFF,
            8 : cls.EUCLIDEAN,
            9 : cls.HAMMING,
            10: cls.JACCARD,
            11: cls.JENSENSHANNON,
            12: cls.KULCZYNSKI1,
            13: cls.KULSINSKI,
            14: cls.MAHALANOBIS,
            15: cls.MINKOWSKI,
            16: cls.ROGERSTANIMOTO,
            17: cls.RUSSELLRAO,
            18: cls.SEUCLIDEAN,
            19: cls.SOKALMICHENER,
            20: cls.SOKALSNEATH,
            21: cls.SQEUCLIDEAN,
            22: cls.YULE,
        }
    
    @classmethod
    def from_str(cls, value: str) -> DistanceMetric:
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> DistanceMetric:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: DistanceMetricType) -> DistanceMetric | None:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """
        if isinstance(value, DistanceMetric):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class InterpolationMode(core.Enum):
    
    BICUBIC       = "bicubic"
    BILINEAR      = "bilinear"
    NEAREST       = "nearest"
    # For PIL compatibility
    BOX           = "box"
    HAMMING       = "hamming"
    LANCZOS       = "lanczos"
    # For OpenCV compatibility
    AREA          = "area"
    CUBIC         = "cubic"
    LANCZOS4      = "lanczos4"
    LINEAR        = "linear"
    LINEAR_EXACT  = "linear_exact"
    MAX           = "max"
    NEAREST_EXACT = "nearest_exact"

    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
        return {
            "bicubic"      : cls.BICUBIC,
            "bilinear"     : cls.BILINEAR,
            "nearest"      : cls.NEAREST,
            "box"          : cls.BOX,
            "hamming"      : cls.HAMMING,
            "lanczos"      : cls.LANCZOS,
            "area"         : cls.AREA,
            "cubic"        : cls.CUBIC,
            "lanczos4"     : cls.LANCZOS4,
            "linear"       : cls.LINEAR,
            "linear_exact" : cls.LINEAR_EXACT,
            "max"          : cls.MAX,
            "nearest_exact": cls.NEAREST_EXACT,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
        return {
            0 : cls.BICUBIC,
            1 : cls.BILINEAR,
            2 : cls.NEAREST,
            3 : cls.BOX,
            4 : cls.HAMMING,
            5 : cls.LANCZOS,
            6 : cls.AREA,
            7 : cls.CUBIC,
            8 : cls.LANCZOS4,
            9 : cls.LINEAR,
            10: cls.LINEAR_EXACT,
            11: cls.MAX,
            12: cls.NEAREST_EXACT,
        }

    @classmethod
    def cv_modes_mapping(cls) -> dict:
        """Returns a dictionary that maps cv2 interpolation mode to the foundation.
        """
        return {
            cls.AREA    : cv2.INTER_AREA,
            cls.CUBIC   : cv2.INTER_CUBIC,
            cls.LANCZOS4: cv2.INTER_LANCZOS4,
            cls.LINEAR  : cv2.INTER_LINEAR,
            cls.MAX     : cv2.INTER_MAX,
            cls.NEAREST : cv2.INTER_NEAREST,
        }

    @classmethod
    def pil_modes_mapping(cls) -> dict:
        """Returns a dictionary that maps PIL interpolation mode to the foundation.
        """
        return {
            cls.NEAREST : 0,
            cls.LANCZOS : 1,
            cls.BILINEAR: 2,
            cls.BICUBIC : 3,
            cls.BOX     : 4,
            cls.HAMMING : 5,
        }
    
    @classmethod
    def from_str(cls, value: str) -> InterpolationMode:
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> InterpolationMode:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: InterpolationModeType) -> InterpolationMode | None:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """
        if isinstance(value, InterpolationMode):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class PaddingMode(core.Enum):
    
    CONSTANT      = "constant"
    # For torch compatibility
    CIRCULAR      = "circular"
    REFLECT       = "reflect"
    REPLICATE     = "replicate"
    # For numpy compatibility
    EDGE          = "edge"
    EMPTY         = "empty"
    LINEAR_RAMP   = "linear_ramp"
    MAXIMUM       = "maximum"
    MEAN          = "mean"
    MEDIAN        = "median"
    MINIMUM       = "minimum"
    SYMMETRIC     = "symmetric"
    WRAP          = "wrap"

    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
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
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
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
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> PaddingMode:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: PaddingModeType) -> PaddingMode | None:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """
        if isinstance(value, PaddingMode):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        return None


class RGB(core.Enum):
    """138 RGB colors."""
    
    MAROON                  = (128,   0,   0)
    DARK_RED                = (139,   0,   0)
    BROWN                   = (165,  42,  42)
    FIREBRICK               = (178,  34,  34)
    CRIMSON                 = (220,  20,  60)
    RED                     = (255,   0,   0)
    TOMATO                  = (255,  99,  71)
    CORAL                   = (255, 127,  80)
    INDIAN_RED              = (205,  92,  92)
    LIGHT_CORAL             = (240, 128, 128)
    DARK_SALMON             = (233, 150, 122)
    SALMON                  = (250, 128, 114)
    LIGHT_SALMON            = (255, 160, 122)
    ORANGE_RED              = (255,  69,   0)
    DARK_ORANGE             = (255, 140,   0)
    ORANGE                  = (255, 165,   0)
    GOLD                    = (255, 215,   0)
    DARK_GOLDEN_ROD         = (184, 134,  11)
    GOLDEN_ROD              = (218, 165,  32)
    PALE_GOLDEN_ROD         = (238, 232, 170)
    DARK_KHAKI              = (189, 183, 107)
    KHAKI                   = (240, 230, 140)
    OLIVE                   = (128, 128,   0)
    YELLOW                  = (255, 255,   0)
    YELLOW_GREEN            = (154, 205,  50)
    DARK_OLIVE_GREEN        = ( 85, 107,  47)
    OLIVE_DRAB              = (107, 142,  35)
    LAWN_GREEN              = (124, 252,   0)
    CHART_REUSE             = (127, 255,   0)
    GREEN_YELLOW            = (173, 255,  47)
    DARK_GREEN              = (  0, 100,   0)
    GREEN                   = (  0, 128,   0)
    FOREST_GREEN            = ( 34, 139,  34)
    LIME                    = (  0, 255,   0)
    LIME_GREEN              = ( 50, 205,  50)
    LIGHT_GREEN             = (144, 238, 144)
    PALE_GREEN              = (152, 251, 152)
    DARK_SEA_GREEN          = (143, 188, 143)
    MEDIUM_SPRING_GREEN     = (  0, 250, 154)
    SPRING_GREEN            = (  0, 255, 127)
    SEA_GREEN               = ( 46, 139,  87)
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_SEA_GREEN        = ( 60, 179, 113)
    LIGHT_SEA_GREEN         = ( 32, 178, 170)
    DARK_SLATE_GRAY         = ( 47,  79,  79)
    TEAL                    = (  0, 128, 128)
    DARK_CYAN               = (  0, 139, 139)
    AQUA                    = (  0, 255, 255)
    CYAN                    = (  0, 255, 255)
    LIGHT_CYAN              = (224, 255, 255)
    DARK_TURQUOISE          = (  0, 206, 209)
    TURQUOISE               = ( 64, 224, 208)
    MEDIUM_TURQUOISE        = ( 72, 209, 204)
    PALE_TURQUOISE          = (175, 238, 238)
    AQUA_MARINE             = (127, 255, 212)
    POWDER_BLUE             = (176, 224, 230)
    CADET_BLUE              = ( 95, 158, 160)
    STEEL_BLUE              = ( 70, 130, 180)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    DEEP_SKY_BLUE           = (  0, 191, 255)
    DODGER_BLUE             = ( 30, 144, 255)
    LIGHT_BLUE              = (173, 216, 230)
    SKY_BLUE                = (135, 206, 235)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    MIDNIGHT_BLUE           = ( 25,  25, 112)
    NAVY                    = (  0,   0, 128)
    DARK_BLUE               = (  0,   0, 139)
    MEDIUM_BLUE             = (  0,   0, 205)
    BLUE                    = (  0,   0, 255)
    ROYAL_BLUE              = ( 65, 105, 225)
    BLUE_VIOLET             = (138,  43, 226)
    INDIGO                  = ( 75,   0, 130)
    DARK_SLATE_BLUE         = ( 72,  61, 139)
    SLATE_BLUE              = (106,  90, 205)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_PURPLE           = (147, 112, 219)
    DARK_MAGENTA            = (139,   0, 139)
    DARK_VIOLET             = (148,   0, 211)
    DARK_ORCHID             = (153,  50, 204)
    MEDIUM_ORCHID           = (186,  85, 211)
    PURPLE                  = (128,   0, 128)
    THISTLE                 = (216, 191, 216)
    PLUM                    = (221, 160, 221)
    VIOLET                  = (238, 130, 238)
    MAGENTA                 = (255,   0, 255)
    ORCHID                  = (218, 112, 214)
    MEDIUM_VIOLET_RED       = (199,  21, 133)
    PALE_VIOLET_RED         = (219, 112, 147)
    DEEP_PINK               = (255,  20, 147)
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
    SADDLE_BROWN            = (139,  69,  19)
    SIENNA                  = (160,  82,  45)
    CHOCOLATE               = (210, 105,  30)
    PERU                    = (205, 133,  63)
    SANDY_BROWN             = (244, 164,  96)
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
    BLACK                   = (  0,   0,   0)
    DIM_GRAY                = (105, 105, 105)
    GRAY                    = (128, 128, 128)
    DARK_GRAY               = (169, 169, 169)
    SILVER                  = (192, 192, 192)
    LIGHT_GRAY              = (211, 211, 211)
    GAINSBORO               = (220, 220, 220)
    WHITE_SMOKE             = (245, 245, 245)
    WHITE                   = (255, 255, 255)


class VisionBackend(core.Enum):
    
    CV      = "cv"
    FFMPEG  = "ffmpeg"
    LIBVIPS = "libvips"
    PIL     = "pil"
    
    @classmethod
    def str_mapping(cls) -> dict:
        """Returns a dictionary that maps strings to the corresponding foundation."""
        return {
            "cv"     : cls.CV,
            "ffmpeg" : cls.FFMPEG,
            "libvips": cls.LIBVIPS,
            "pil"    : cls.PIL,
        }
    
    @classmethod
    def int_mapping(cls) -> dict:
        """Returns a dictionary that maps integers to the foundation."""
        return {
            0: cls.CV,
            1: cls.FFMPEG,
            2: cls.LIBVIPS,
            3: cls.PIL,
        }
    
    @classmethod
    def from_str(cls, value: str) -> VisionBackend:
        """Converts a string to an foundation.
        
        Args:
            value: The string to convert to an foundation.
        
        Returns:
            The foundation.
        """
        assert value.lower() in cls.str_mapping()
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> VisionBackend:
        """Converts an integer to an foundation.
        
        Args:
            value: The int value to be converted to an foundation.
        
        Returns:
            The foundation.
        """
        assert value in cls.int_mapping()
        return cls.int_mapping()[value]
    
    @classmethod
    def from_value(cls, value: VisionBackendType) -> VisionBackend:
        """Converts an arbitrary value to an foundation.
        
        Args:
            value: The value to be converted.
        
        Returns:
            The foundation.
        """

        if isinstance(value, VisionBackend):
            return value
        if isinstance(value, int):
            return cls.from_int(value)
        if isinstance(value, str):
            return cls.from_str(value)
        return VISION_BACKEND


VISION_BACKEND = VisionBackend.CV

# endregion
