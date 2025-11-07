#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains various enumeration classes ."""

__all__ = [
    "AppleRGB",
    "BBoxFormat",
    "BasicRGB",
    "ConfigExtension",
    "DepthSource",
    "Enum",
    "ImageExtension",
    "InfraredSource",
    "MLType",
    "MemoryUnit",
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


# ----- Base -----
class Enum(enum.Enum):
    """An extension of ``enum.Enum`` with additional utilities."""
    
    @classmethod
    def __init_subclass__(cls):
        """Initialize the set of values when the subclass is created."""
        cls._names         = list(cls)
        cls._values        = [member.value for member in cls]
        cls._int_to_enum   = {i: member for i, member in enumerate(cls)}
        cls._value_to_enum = {member.value: member for member in cls}
        cls._str_to_enum   = {str(member.name).lower(): member for member in cls}
        
    @classmethod
    def __contains__(cls, value: Any) -> bool:
        """Checks if the current enum class contains a specific value.

        Args:
            value: Value to check.

        Returns:
            ``True`` if ``value`` is in the enum, ``False`` otherwise.
        
        Notes:
            Usage: ``if value in EnumClass: ...``
        """
        return value in cls or value in cls._values
    
    @classmethod
    def random(cls):
        """Returns a random enum member."""
        return random.choice(list(cls))
    
    @classmethod
    def random_value(cls):
        """Returns a random enum value."""
        return cls.random().value
    
    @classmethod
    def names(cls) -> list:
        """Returns a ``list`` of all enum members."""
        return cls._names
    
    @classmethod
    def values(cls) -> list[Any]:
        """Returns a ``list`` of all enum values."""
        return cls._values
    
    @classmethod
    def int_to_enum(cls) -> dict:
        """Creates a ``dict`` mapping Enum indexes to Enum members."""
        return cls._int_to_enum
    
    @classmethod
    def value_to_enum(cls) -> dict:
        """Create a ``dict`` mapping Enum values to Enum members."""
        return cls._value_to_enum
    
    @classmethod
    def str_to_enum(cls) -> dict:
        """Create a ``dict`` mapping Enum names (lowercase) to Enum members."""
        return cls._str_to_enum
    
    @classmethod
    def from_str(cls, a_str: str):
        """Create an Enum member from a ``str``.
         
         Raises:
            ValueError: If ``a_str`` is not a valid Enum name.
        """
        str_to_enum = cls.str_to_enum()
        value_lower = a_str.lower()
        if value_lower not in str_to_enum:
            raise ValueError(f"``a_str`` must be one of {str_to_enum}, got {value_lower}.")
        return str_to_enum[value_lower]
    
    @classmethod
    def from_int(cls, an_int: int):
        """Create an Enum member from an ``int`` index.
    
        Raises:
            ValueError: If ``an_int`` is not a valid Enum index.
        """
        int_to_enum = cls.int_to_enum()
        if an_int not in int_to_enum:
            raise ValueError(f"``an_int`` must be one of {int_to_enum}, got {an_int}.")
        return int_to_enum[an_int]
    
    @classmethod
    def from_value(cls, value: Any):
        """Create an Enum member from an arbitrary ``value``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        raise TypeError(f"``value`` must be a str or int, got {type(value)}.")


# ----- Color -----
class RGB(Enum):
    """138 RGB colors."""
    
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


class AppleRGB(Enum):
    """Apple's RGB colors."""
    
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


class BasicRGB(Enum):
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


# ----- Device -----
class MemoryUnit(Enum):
    """Memory units."""
    
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
    def name_to_byte(cls):
        """Create a ``dict`` mapping memory units to their corresponding number
        of bytes.
        """
        return {
            cls.B : 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }
    

# ----- File -----
class ConfigExtension(Enum):
    """Configuration file extensions."""
    
    CFG    = ".cfg"
    CONFIG = ".config"
    JSON   = ".json"
    NAMES  = ".names"
    PY     = ".py"
    TXT    = ".txt"
    YAML   = ".yaml"
    YML    = ".yml"


class ImageExtension(Enum):
    """Image file extensions."""
    
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
    """Video file extensions."""
    
    AVI  = ".avi"
    M4V  = ".m4v"
    MKV  = ".mkv"
    MOV  = ".mov"
    MP4  = ".mp4"
    MPEG = ".mpeg"
    MPG  = ".mpg"
    WMV  = ".wmv"


class WeightExtension(Enum):
    """Weight file extensions."""

    CKPT    = ".ckpt"
    ONNX    = ".onnx"
    PT      = ".pt"
    PTH     = ".pth"
    TAR     = ".tar"
    WEIGHTS = ".weights"


# ----- ML/DL -----
class MLType(Enum):
    """Machine learning types."""
    
    INFERENCE       = "inference"        # Inference Only: we don't have training code.
    TRADITIONAL     = "traditional"      # Traditional Method (non-learning).
    SUPERVISED      = "supervised"       # Supervised learning with labeled data.
    UNSUPERVISED    = "unsupervised"     # Unsupervised learning with unlabeled data.
    SELF_SUPERVISED = "self_supervised"  # Self-Supervised learning with self-generated supervision.
    ZERO_SHOT       = "zero_shot"        # Zero-Shot learning without any training data.
    
    @classmethod
    def trainable(cls) -> list:
        """Returns a ``list`` of conventional training machine learning types."""
        return [cls.SELF_SUPERVISED, cls.SUPERVISED, cls.UNSUPERVISED]


class RunMode(Enum):
    """Run modes."""
    
    TRAIN   = "train"
    PREDICT = "predict"
    METRIC  = "metric"


class Split(Enum):
    """Dataset's split types."""
    
    TRAIN   = "train"
    VAL     = "val"
    TEST    = "test"
    PREDICT = "predict"


class TRTPrecision(Enum):
    """TensorRT precision modes."""
    
    FP32    = "fp32"                    # 32-bit floating point
    FP16    = "fp16"                    # 16-bit floating point
    FP16N32 = "fp16n32"                 # 16-bit floating point with 32-bit normalization
    FP8     = "fp8"                     # 8-bit floating point
    INT8    = "int8"                    # 8-bit integer


# ----- Task -----
class Task(Enum):
    """Task types."""
    
    # ----- Generative AI -----
    # Image Generation
    IMG2IMG     = "img2img"             # Image-to-Image Translation
    
    # ----- Computer Vision -----
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


# ----- Vision -----
class BBoxFormat(Enum):
    """Bounding box formats."""

    # Format
    XYWH       = "xywh"                 # COCO format: [ x,  y,  w,  h]
    XYXY       = "xyxy"                 # VOC  format: [x1, y1, x2, y2]
    CXCYWHN    = "cxcywhn"              # YOLO format: [cx, cy,  w,  h] normalized
    COCO       = "coco"
    VOC        = "voc"
    YOLO       = "yolo"
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
        """Returns a ``list`` of all bounding box formats."""
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
        """Returns a ``list`` of all bounding box format conversion codes."""
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


class DepthSource(Enum):
    """Depth data input sources."""

    DAAC_ViTS = "depth_daac_vits"       # Depth Anything at Any Condition with ViT-S encoder
    DAv2_ViTB = "depth_dav2_vitb"       # Depth Anything v2 with ViT-B encoder
    DAv2_ViTL = "depth_dav2_vitl"       # Depth Anything v2 with ViT-L encoder
    DAv2_ViTS = "depth_dav2_vits"       # Depth Anything v2 with ViT-S encoder
    DEPTH_PRO = "depth_pro"             # Depth Pro
    DEPTH     = "depth"


class InfraredSource(Enum):
    """Infrared data input sources."""
    
    INFRARED = "infrared"


class TrackState(Enum):
    """Enumeration type for a single target track state.
    
    Newly created tracks are classified as ``NEW`` until enough evidence has been
    collected. Then, the track state is changed to ``TRACKED``. Tracks that are no
    longer alive are classified as ``REMOVED`` to mark them for removal from the set of
    active tracks.
    """
    
    NEW      = 0
    TRACKED  = 1
    LOST     = 2
    REMOVED  = 3
    REPLACED = 4
    COUNTED  = 5
