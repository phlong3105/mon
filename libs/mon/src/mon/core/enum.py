#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Enumeration helpers and project-specific enumerations.

This module provides an extended Enum base class with lookup and construction
utilities, as well as a collection of project-specific enums for colors,
formats, tasks, memory units, and more to enable consistent identifiers and
conversions across the codebase.
"""

__all__ = [
    "ActiveLearningPhase",
    "AppleRGB",
    "BBoxFormat",
    "ConfigExtension",
    "DepthSource",
    "Enum",
    "ImageExtension",
    "InfraredSource",
    "MLType",
    "MemoryUnit",
    "RGB",
    "RGB12",
    "RunMode",
    "Split",
    "TRTPrecision",
    "Task",
    "TrackState",
    "VideoExtension",
    "WeightExtension",
]

import enum
import random
from typing import Any


# ==============================================================================
# CORE INFRASTRUCTURE
# ==============================================================================

# --- Metaclasses (The CustomEnumMeta logic) ---
class CustomEnumMeta(enum.EnumMeta):
    """A metaclass for flexible enum construction.

    Enable flexible enum construction so subclasses accept names or indices
    when constructing members.
    """

    def __call__(cls, value: Any, *args, **kwargs):
        """Construct or convert a value into an enum member.

        Accept the usual Enum construction calls and delegate single-argument
        conversions to from_value for convenience.
        """
        if args or kwargs:
            # Fallback for other unexpected calls
            return super().__call__(value, *args, **kwargs)

        return cls.from_value(value)


# --- Base Classes (The extended Enum class) ---
class Enum(enum.Enum, metaclass=CustomEnumMeta):
    """An extended enum with convenience utilities.

    Add caching, random selection, and conversion helpers for names, indices,
    and values to simplify common enum operations.

    Attributes:
        _names (list[Enum]): Cached list of enum members in declaration order.
        _values (list[Any]): Cached list of enum values in declaration order.
        _int_to_enum (dict[int, Enum]): Mapping from integer indices to members.
        _value_to_enum (dict[Any, Enum]): Mapping from values to members.
        _str_to_enum (dict[str, Enum]): Mapping from lowercase names to members.
    """

    @classmethod
    def __init_subclass__(cls):
        """Initialize cached helper mappings on subclass definition."""
        cls._names         = list(cls)
        cls._values        = [member.value for member in cls]
        cls._int_to_enum   = {i: member for i, member in enumerate(cls)}
        cls._value_to_enum = {member.value: member for member in cls}
        cls._str_to_enum   = {str(member.name).lower(): member for member in cls}

    @classmethod
    def __contains__(cls, value: Any) -> bool:
        """Return True if the value is represented by the enum."""
        return value in cls or value in cls._values

    @classmethod
    def random(cls):
        """Return a random enum member.

        Choose one member uniformly at random.
        """
        return random.choice(list(cls))

    @classmethod
    def random_value(cls):
        """Return the value of a random enum member.

        This is equivalent to calling random().value.
        """
        return cls.random().value

    @classmethod
    def names(cls) -> list:
        """Return a list of all enum members.

        Preserve declaration order.
        """
        return cls._names

    @classmethod
    def values(cls) -> list[Any]:
        """Return a list of all enum values.

        Preserve declaration order.
        """
        return cls._values

    @classmethod
    def int_to_enum(cls) -> dict:
        """Return a mapping from integer indices to enum members.

        Indices correspond to declaration order, starting at zero.
        """
        return cls._int_to_enum

    @classmethod
    def value_to_enum(cls) -> dict:
        """Return a mapping from enum values to enum members."""
        return cls._value_to_enum

    @classmethod
    def str_to_enum(cls) -> dict:
        """Return a mapping from lowercase member names to enum members."""
        return cls._str_to_enum

    # --- Initialize ---
    @classmethod
    def from_str(cls, a_str: str):
        """Convert a name string to an enum member.

        Args:
            a_str: Member name (case-insensitive).

        Raises:
            ValueError: If ``a_str`` is not valid.
        """
        str_to_enum = cls.str_to_enum()
        value_lower = a_str.lower()
        if value_lower not in str_to_enum:
            raise ValueError(f"``a_str`` must be one of {list(str_to_enum)}, got {value_lower}.")
        return str_to_enum[value_lower]

    @classmethod
    def from_int(cls, an_int: int):
        """Convert an integer index to an enum member.

        Args:
            an_int: Index corresponding to declaration order.

        Raises:
            ValueError: If ``an_int`` is out of range.
        """
        int_to_enum = cls.int_to_enum()
        if an_int not in int_to_enum:
            raise ValueError(f"``an_int`` must be one of {list(int_to_enum)}, got {an_int}.")
        return int_to_enum[an_int]

    @classmethod
    def from_value(cls, value: Any):
        """Convert a supported input into an enum member.

        Args:
            value: Enum member, name, or index.

        Raises:
            TypeError: If ``value``'s type is unsupported.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        raise TypeError(f"``value`` must be a str or int, got {type(value)}.")


# ==============================================================================
# VISUAL & UI DOMAIN
# ==============================================================================

# --- Color Palettes ---
class RGB(Enum):
    """An enumeration of standard RGB colors.

    Provide (R, G, B) tuples for named colors.

    Attributes:
        value (tuple[int, int, int]): RGB tuple for the color.
    """
    
    ALICE_BLUE              = (240, 248, 255)
    ANTIQUE_WHITE           = (250, 235, 215)
    AQUA                    = (  0, 255, 255)
    AQUA_MARINE             = (127, 255, 212)
    AZURE                   = (240, 255, 255)
    BEIGE                   = (245, 245, 220)
    BISQUE                  = (255, 228, 196)
    BLACK                   = (  0,   0,   0)
    BLANCHED_ALMOND         = (255, 235, 205)
    BLUE                    = (  0,   0, 255)
    BLUE_VIOLET             = (138,  43, 226)
    BROWN                   = (165,  42,  42)
    BURLY_WOOD              = (222, 184, 135)
    CADET_BLUE              = ( 95, 158, 160)
    CHART_REUSE             = (127, 255,   0)
    CHOCOLATE               = (210, 105,  30)
    CORAL                   = (255, 127,  80)
    CORN_FLOWER_BLUE        = (100, 149, 237)
    CORN_SILK               = (255, 248, 220)
    CRIMSON                 = (220,  20,  60)
    CYAN                    = (  0, 255, 255)
    DARK_BLUE               = (  0,   0, 139)
    DARK_CYAN               = (  0, 139, 139)
    DARK_GOLDEN_ROD         = (184, 134,  11)
    DARK_GRAY               = (169, 169, 169)
    DARK_GREEN              = (  0, 100,   0)
    DARK_KHAKI              = (189, 183, 107)
    DARK_MAGENTA            = (139,   0, 139)
    DARK_OLIVE_GREEN        = ( 85, 107,  47)
    DARK_ORANGE             = (255, 140,   0)
    DARK_ORCHID             = (153,  50, 204)
    DARK_RED                = (139,   0,   0)
    DARK_SALMON             = (233, 150, 122)
    DARK_SEA_GREEN          = (143, 188, 143)
    DARK_SLATE_BLUE         = ( 72,  61, 139)
    DARK_SLATE_GRAY         = ( 47,  79,  79)
    DARK_TURQUOISE          = (  0, 206, 209)
    DARK_VIOLET             = (148,   0, 211)
    DEEP_PINK               = (255,  20, 147)
    DEEP_SKY_BLUE           = (  0, 191, 255)
    DIM_GRAY                = (105, 105, 105)
    DODGER_BLUE             = ( 30, 144, 255)
    FIREBRICK               = (178,  34,  34)
    FLORAL_WHITE            = (255, 250, 240)
    FOREST_GREEN            = ( 34, 139,  34)
    GAINSBORO               = (220, 220, 220)
    GHOST_WHITE             = (248, 248, 255)
    GOLD                    = (255, 215,   0)
    GOLDEN_ROD              = (218, 165,  32)
    GRAY                    = (128, 128, 128)
    GREEN                   = (  0, 128,   0)
    GREEN_YELLOW            = (173, 255,  47)
    HONEYDEW                = (240, 255, 240)
    HOT_PINK                = (255, 105, 180)
    INDIAN_RED              = (205,  92,  92)
    INDIGO                  = ( 75,   0, 130)
    IVORY                   = (255, 255, 240)
    KHAKI                   = (240, 230, 140)
    LAVENDER                = (230, 230, 250)
    LAVENDER_BLUSH          = (255, 240, 245)
    LAWN_GREEN              = (124, 252,   0)
    LEMON_CHIFFON           = (255, 250, 205)
    LIGHT_BLUE              = (173, 216, 230)
    LIGHT_CORAL             = (240, 128, 128)
    LIGHT_CYAN              = (224, 255, 255)
    LIGHT_GOLDEN_ROD_YELLOW = (250, 250, 210)
    LIGHT_GRAY              = (211, 211, 211)
    LIGHT_GREEN             = (144, 238, 144)
    LIGHT_PINK              = (255, 182, 193)
    LIGHT_SALMON            = (255, 160, 122)
    LIGHT_SEA_GREEN         = ( 32, 178, 170)
    LIGHT_SKY_BLUE          = (135, 206, 250)
    LIGHT_SLATE_GRAY        = (119, 136, 153)
    LIGHT_STEEL_BLUE        = (176, 196, 222)
    LIGHT_YELLOW            = (255, 255, 224)
    LIME                    = (  0, 255,   0)
    LIME_GREEN              = (50 , 205,  50)
    LINEN                   = (250, 240, 230)
    MAGENTA                 = (255,   0, 255)
    MAROON                  = (128,   0,   0)
    MEDIUM_AQUA_MARINE      = (102, 205, 170)
    MEDIUM_BLUE             = (  0,   0, 205)
    MEDIUM_ORCHID           = (186,  85, 211)
    MEDIUM_PURPLE           = (147, 112, 219)
    MEDIUM_SEA_GREEN        = ( 60, 179, 113)
    MEDIUM_SLATE_BLUE       = (123, 104, 238)
    MEDIUM_SPRING_GREEN     = (  0, 250, 154)
    MEDIUM_TURQUOISE        = ( 72, 209, 204)
    MEDIUM_VIOLET_RED       = (199,  21, 133)
    MIDNIGHT_BLUE           = ( 25,  25, 112)
    MINT_CREAM              = (245, 255, 250)
    MISTY_ROSE              = (255, 228, 225)
    MOCCASIN                = (255, 228, 181)
    NAVAJO_WHITE            = (255, 222, 173)
    NAVY                    = (  0,   0, 128)
    OLD_LACE                = (253, 245, 230)
    OLIVE                   = (128, 128,   0)
    OLIVE_DRAB              = (107, 142,  35)
    ORANGE                  = (255, 165,   0)
    ORANGE_RED              = (255,  69,   0)
    ORCHID                  = (218, 112, 214)
    PALE_GOLDEN_ROD         = (238, 232, 170)
    PALE_GREEN              = (152, 251, 152)
    PALE_TURQUOISE          = (175, 238, 238)
    PALE_VIOLET_RED         = (219, 112, 147)
    PAPAYA_WHIP             = (255, 239, 213)
    PEACH_PUFF              = (255, 218, 185)
    PERU                    = (205, 133, 63)
    PINK                    = (255, 192, 203)
    PLUM                    = (221, 160, 221)
    POWDER_BLUE             = (176, 224, 230)
    PURPLE                  = (128,   0, 128)
    RED                     = (255,   0,   0)
    ROSY_BROWN              = (188, 143, 143)
    ROYAL_BLUE              = ( 65, 105, 225)
    SADDLE_BROWN            = (139,  69,  19)
    SALMON                  = (250, 128, 114)
    SANDY_BROWN             = (244, 164,  96)
    SEA_GREEN               = ( 46, 139,  87)
    SEA_SHELL               = (255, 245, 238)
    SIENNA                  = (160,  82,  45)
    SILVER                  = (192, 192, 192)
    SKY_BLUE                = (135, 206, 235)
    SLATE_BLUE              = (106,  90, 205)
    SLATE_GRAY              = (112, 128, 144)
    SNOW                    = (255, 250, 250)
    SPRING_GREEN            = (  0, 255, 127)
    STEEL_BLUE              = ( 70, 130, 180)
    TAN                     = (210, 180, 140)
    TEAL                    = (  0, 128, 128)
    THISTLE                 = (216, 191, 216)
    TOMATO                  = (255,  99,  71)
    TURQUOISE               = ( 64, 224, 208)
    VIOLET                  = (238, 130, 238)
    WHEAT                   = (245, 222, 179)
    WHITE                   = (255, 255, 255)
    WHITE_SMOKE             = (245, 245, 245)
    YELLOW                  = (255, 255,   0)
    YELLOW_GREEN            = (154, 205,  50)


class RGB12(Enum):
    """A small basic RGB palette.

    Provide a compact set of common RGB tuples for simple palettes.

    Attributes:
        value (tuple[int, int, int]): RGB tuple for the color.
    """
    
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


class AppleRGB(Enum):
    """An Apple UI color palette.

    Provide named Apple-specific RGB tuples commonly used in UI palettes.

    Attributes:
        value (tuple[int, int, int]): RGB tuple for the color.
    """
    
    BLACK       = (  0,   0,   0)
    BLUE        = (  0, 122, 255)
    BROWN       = (162, 132,  94)
    CYAN        = ( 50, 173, 230)
    GRAY        = (128, 128, 128)
    GRAY2       = (174, 174, 178)
    GRAY3       = (199, 199, 204)
    GRAY4       = (209, 209, 214)
    GRAY5       = (229, 229, 234)
    GRAY6       = (242, 242, 247)
    GREEN       = ( 52, 199,  89)
    INDIGO      = ( 85, 190, 240)
    MINT        = (  0, 199,  89)
    ORANGE      = (255, 149,   5)
    PINK        = (255,  45,  85)
    PURPLE      = ( 88,  86, 214)
    RED         = (255,  59,  48)
    TEAL        = ( 90, 200, 250)
    WHITE       = (255, 255, 255)
    YELLOW      = (255, 204,   0)
    DARK_BLUE   = (  0,  64, 221)
    DARK_BROWN  = (127, 101,  69)
    DARK_CYAN   = (  0, 113, 164)
    DARK_GRAY2  = ( 99,  99, 102)
    DARK_GRAY3  = ( 72,  72,  74)
    DARK_GRAY4  = ( 58,  58,  60)
    DARK_GRAY5  = ( 44,  44,  46)
    DARK_GRAY6  = ( 28,  28,  30)
    DARK_GREEN  = ( 36, 138,  61)
    DARK_INDIGO = ( 54,  52, 163)
    DARK_MINT   = ( 12, 129, 123)
    DARK_ORANGE = (201,  52,   0)
    DARK_PINK   = (211,  15,  69)
    DARK_PURPLE = (137,  68, 171)
    DARK_RED    = (255,  69,  58)
    DARK_TEAL   = (  0, 130, 153)
    DARK_YELLOW = (178,  80,   0)


# --- Units ---
class MemoryUnit(Enum):
    """A memory unit enumeration.

    Provide names for common memory units and helpers to convert to bytes.

    Attributes:
        value (str): Unit string such as "B", "KB", "MB", etc.
    """
    
    B  = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"

    '''
    @classmethod
    def str_to_enum(cls) -> dict:
        """Return a dictionary mapping strings to ``MemoryUnit``.

        This method provides a mapping from string representations of memory units
        to their corresponding ``MemoryUnit`` enum values. This is useful for
        converting string inputs to enum values in a consistent manner.

        Returns:
            A dictionary where the keys are string representations of memory
            units and the values are the corresponding ``MemoryUnit`` enum values.
        """
        return {
            "b" : cls.B,
            "kb": cls.KB,
            "mb": cls.MB,
            "gb": cls.GB,
            "tb": cls.TB,
            "pb": cls.PB,
        }
    '''

    @classmethod
    def name_to_byte(cls) -> dict:
        """Return mapping of this enum to byte multipliers.

        Map each enum member to the number of bytes represented by one unit.
        """
        return {
            cls.B : 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }


# ==============================================================================
# FILESYSTEM & IO DOMAIN
# ==============================================================================

# --- Extensions ---
class ConfigExtension(Enum):
    """A set of configuration file extensions.

    Provide common config filename suffixes including the leading dot.

    Attributes:
        value (str): File extension string including the leading dot.
    """
    
    CFG    = ".cfg"
    CONFIG = ".config"
    JSON   = ".json"
    NAMES  = ".names"
    PY     = ".py"
    TXT    = ".txt"
    YAML   = ".yaml"
    YML    = ".yml"


class ImageExtension(Enum):
    """A set of common image file extensions.

    Provide common image file suffixes including the leading dot.

    Attributes:
        value (str): Image file extension including the leading dot.
    """
    
    ARW  = ".arw"
    BMP  = ".bmp"
    DNG  = ".dng"
    JPEG = ".jpeg"
    JPG  = ".jpg"
    PNG  = ".png"
    PPM  = ".ppm"
    RAF  = ".raf"
    TIF  = ".tif"
    TIFF = ".tiff"


class VideoExtension(Enum):
    """A set of common video file extensions.

    Provide common video file suffixes including the leading dot.

    Attributes:
        value (str): Video file extension including the leading dot.
    """
    
    AVI  = ".avi"
    M4V  = ".m4v"
    MKV  = ".mkv"
    MOV  = ".mov"
    MP4  = ".mp4"
    MPEG = ".mpeg"
    MPG  = ".mpg"
    WMV  = ".wmv"


class WeightExtension(Enum):
    """A set of model weight file extensions.

    Provide typical suffixes used for model checkpoints and weights.

    Attributes:
        value (str): Weight/checkpoint file extension including the dot.
    """
    
    CKPT    = ".ckpt"
    ONNX    = ".onnx"
    PT      = ".pt"
    PTH     = ".pth"
    TAR     = ".tar"
    WEIGHTS = ".weights"


# ==============================================================================
# MACHINE LEARNING & WORKFLOW DOMAIN
# ==============================================================================

# --- Orchestration (RunMode, Split, ActiveLearningPhase) ---
class RunMode(Enum):
    """A set of pipeline run modes.

    Indicate whether the code is running training, prediction, or metrics.

    Attributes:
        value (str): String identifier for the run mode.
    """
    
    TRAIN   = "train"
    PREDICT = "predict"
    METRIC  = "metric"


class Split(Enum):
    """A set of dataset split identifiers.

    Represent dataset subsets such as train, val, test, predict.

    Attributes:
        value (str): String identifier for the dataset split.
    """
    
    TRAIN   = "train"
    VAL     = "val"
    TEST    = "test"
    PREDICT = "predict"


class ActiveLearningPhase(Enum):
    """A set of active learning workflow phases.

    Enumerate the discrete experiment workflow stages.

    Attributes:
        value (str): String identifier for the phase.
    """
    
    TRAINING         = "training"
    METROLOGY        = "metrology"
    QUERY            = "query"
    LABELING         = "labeling"
    DATA_INTEGRATION = "data_integration"
    
    
# --- Paradigms (MLType, Task) ---
class Task(Enum):
    """A set of supported task identifiers.

    Enumerate the high-level tasks that models in the project implement.

    Attributes:
        value (str): String identifier for the task.
    """
    
    # --- Generative AI ---
    # Image Generation
    IMG2IMG     = "img2img"             # Image-to-Image Translation
    
    # --- Computer Vision ---
    # Enhancement
    AWB         = "awb"                 # Auto White Balance
    COLORIZE    = "colorize"            # Image Colorization
    EXPOSURE    = "exposure"            # Exposure Correction + Automatic Exposure
    ISP         = "isp"                 # Image Signal Processing
    LLE         = "lle"                 # Low-Light Enhancement
    MEF         = "mef"                 # Multi-Exposure Fusion
    NTE         = "nte"                 # Night-Time Enhancement
    RETOUCH     = "retouch"             # Retouching
    UWE         = "underwater"          # UnderWater Enhancement
    # Restoration
    DEBAND      = "deband"              # Debanding
    DEBLUR      = "deblur"              # Deblurring
    DEFLARE     = "deflare"             # Deflaring
    DEHAZE      = "dehaze"              # Dehazing
    DENOISE     = "denoise"             # Denoising
    DERAIN      = "derain"              # Deraining
    DESNOW      = "desnow"              # Desnowing
    INPAINT     = "inpaint"             # Inpainting
    SR          = "sr"                  # Super-Resolution
    # High-Level Vision
    BGSUBTRACT  = "bgsubtract"          # Background Subtraction
    CLASSIFY    = "classify"            # Classification
    DETECT      = "detect"              # Object Detection
    MONODEPTH   = "mono_depth"          # Monocular-Depth Estimation
    OBB         = "obb"                 # Oriented-Bounding Box Detection
    POSE        = "pose"                # Pose Estimation
    SEGMENT     = "segment"             # Semantic Segmentation
    TRACK       = "track"               # Tracking
    VIDEO       = "video"               # Video Processing


class MLType(Enum):
    """A set of machine learning approach types.

    Categorize models by their ML paradigm.

    Attributes:
        value (str): String identifier for the ML type.
    """

    INFERENCE       = "inference"        # Inference Only: we don't have training code.
    TRADITIONAL     = "traditional"      # Traditional Method (non-learning).
    SUPERVISED      = "supervised"       # Supervised learning with labeled data.
    UNSUPERVISED    = "unsupervised"     # Unsupervised learning with unlabeled data.
    SELF_SUPERVISED = "self_supervised"  # Self-Supervised (or Semi-Supervised) learning with self-generated supervision.
    ZERO_SHOT       = "zero_shot"        # Zero-Shot learning without any training data.

    @classmethod
    def trainable(cls) -> list:
        """Return ML types that are trainable.

        Return MLType members suitable for training.
        """
        return [cls.SELF_SUPERVISED, cls.SUPERVISED, cls.UNSUPERVISED]


# --- Optimization (TRTPrecision) ---
class TRTPrecision(Enum):
    """A set of TensorRT numeric precision modes.

    Specify desired precision for TensorRT optimizations.

    Attributes:
        value (str): String identifier for the precision mode.
    """
    
    FP32    = "fp32"     # 32-bit floating point
    FP16    = "fp16"     # 16-bit floating point
    FP16N32 = "fp16n32"  # 16-bit floating point with 32-bit normalization
    FP8     = "fp8"      # 8-bit floating point
    INT8    = "int8"     # 8-bit integer


# ==============================================================================
#COMPUTER VISION DOMAIN
# ==============================================================================

# --- Spatial Formats ---
class BBoxFormat(Enum):
    """A set of bounding box formats and conversion codes.

    Include format identifiers and conversion code members.

    Attributes:
        value (str): String identifier for the bbox format or conversion.
    """

    # Format
    XYWH         = "xywh"               # COCO format: [ x,  y,  w,  h]
    XYXY         = "xyxy"               # VOC  format: [x1, y1, x2, y2]
    CXCYWHN      = "cxcywhn"            # YOLO format: [cx, cy,  w,  h] normalized
    COCO         = "coco"
    VOC          = "voc"
    YOLO         = "yolo"
    # Format conversion
    XYWH2XYXY    = "xywh_to_xyxy"       # Convert from COCO to VOC
    XYWH2CXCYWHN = "xywh_to_cxcywhn"    # Convert from COCO to YOLO
    XYXY2XYWH    = "xyxy_to_xywh"       # Convert from VOC  to COCO
    XYXY2CXCYWHN = "xyxy_to_cxcywhn"    # Convert from VOC  to YOLO
    CXCYWHN2XYXY = "cxcywhn_to_xyxy"    # Convert from YOLO to VOC
    CXCYWHN2XYWH = "cxcywhn_to_xywh"    # Convert from YOLO to COCO
    COCO2VOC     = "coco_to_voc"        # Convert from COCO to VOC
    COCO2YOLO    = "coco_to_yolo"       # Convert from COCO to YOLO
    VOC2COCO     = "voc_to_coco"        # Convert from VOC  to COCO
    VOC2YOLO     = "voc_to_yolo"        # Convert from VOC  to YOLO
    YOLO2VOC     = "yolo_to_voc"        # Convert from YOLO to VOC
    YOLO2COCO    = "yolo_to_coco"       # Convert from YOLO to COCO

    @classmethod
    def formats(cls) -> list:
        """Return a list of standard bounding box formats.

        Include common format identifiers such as XYXY and XYWH.
        """
        return [
            cls.XYXY,
            cls.XYWH,
            cls.CXCYWHN,
            cls.VOC,
            cls.COCO,
            cls.YOLO,
        ]

    @classmethod
    def conversion_codes(cls) -> list:
        """Return a list of bounding box conversion code members.

        Include members whose values indicate conversion operations.
        """
        return [
            cls.XYXY2XYWH,
            cls.XYXY2CXCYWHN,
            cls.XYWH2XYXY,
            cls.XYWH2CXCYWHN,
            cls.CXCYWHN2XYXY,
            cls.CXCYWHN2XYWH,
            cls.VOC2COCO,
            cls.VOC2YOLO,
            cls.COCO2VOC,
            cls.COCO2YOLO,
            cls.YOLO2VOC,
            cls.YOLO2COCO,
        ]


# --- Data Sources (DepthSource, InfraredSource) ---
class DepthSource(Enum):
    """A set of depth data source identifiers.

    Indicate which model or pipeline produced depth data.

    Attributes:
        value (str): String identifier for the depth source.
    """

    DAAC_ViTS = "depth_daac_vits"       # Depth Anything at Any Condition with ViT-S encoder
    DAv2_ViTB = "depth_dav2_vitb"       # Depth Anything v2 with ViT-B encoder
    DAv2_ViTL = "depth_dav2_vitl"       # Depth Anything v2 with ViT-L encoder
    DAv2_ViTS = "depth_dav2_vits"       # Depth Anything v2 with ViT-S encoder
    DEPTH_PRO = "depth_pro"             # Depth Pro
    DEPTH     = "depth"


class InfraredSource(Enum):
    """A set of infrared data source identifiers.

    Provide identifiers for infrared data sources.

    Attributes:
        value (str): String identifier for the infrared source.
    """
    
    INFRARED = "infrared"


# --- State Management (TrackState) ---
class TrackState(Enum):
    """A set of object tracking lifecycle states.

    Define integer codes representing stages such as NEW, TRACKED, and LOST.

    Attributes:
        value (int): Integer code representing the track state.
    """
    
    NEW      = 0
    TRACKED  = 1
    LOST     = 2
    REMOVED  = 3
    REPLACED = 4
    COUNTED  = 5
