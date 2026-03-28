#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Global Enums.

This module provides global Enums.
"""

from __future__ import annotations

__all__ = [
    "AlbumTargetType",
    "BBoxFormat",
    "ConfigExtension",
    "DeviceType",
    "ImageExtension",
    "MemoryUnit",
    "Precision",
    "RunMode",
    "Split",
    "Task",
    "VideoExtension",
    "WeightExtension",
]

from .base.enum import MultiStrEnum


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Units ---

class MemoryUnit(MultiStrEnum):
    """Enum for memory units."""

    B = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB", "default"
    TB = "TB"
    PB = "PB"

    @classmethod
    def names_to_bytes(cls) -> dict:
        """Return mapping of this enum to byte multipliers."""
        return {
            cls.B: 1024 ** 0,
            cls.KB: 1024 ** 1,
            cls.MB: 1024 ** 2,
            cls.GB: 1024 ** 3,
            cls.TB: 1024 ** 4,
            cls.PB: 1024 ** 5,
        }


# --- File Extensions ---

class ConfigExtension(MultiStrEnum):
    """Enum for configuration file extensions."""

    YAML = ".yaml", ".yml", "default"
    CFG =  ".cfg", ".config"
    JSON = ".json"
    TXT = ".txt"


class ImageExtension(MultiStrEnum):
    """Enum for common image file extensions."""

    ARW = ".arw"
    BMP = ".bmp"
    DNG = ".dng"
    JPG = ".jpg", ".jpeg", "default"
    PNG = ".png"
    PPM = ".ppm"
    RAF = ".raf"
    TIF = ".tif"
    TIFF = ".tiff"


class VideoExtension(MultiStrEnum):
    """Enum for common video file extensions."""

    AVI = ".avi"
    M4V = ".m4v"
    MKV = ".mkv"
    MOV = ".mov"
    MP4 = ".mp4", "default"
    MPEG = ".mpeg"
    MPG = ".mpg"
    WMV = ".wmv"


class WeightExtension(MultiStrEnum):
    """Enum for common model weight file extensions."""

    CKPT = ".ckpt"
    ONNX = ".onnx"
    PT = ".pt", ".pth", "default"
    TAR = ".tar"
    WEIGHTS = ".weights"


# --- Data ---

class BBoxFormat(MultiStrEnum):
    """Enum for bounding box formats and conversion codes."""

    # Formats
    CXCYWHN = "cxcywhn", "yolo", "default"            # YOLO format: [cx, cy, w, h] normalized
    XYWH = "xywh", "coco"                             # COCO format: [x, y, w, h]
    XYXY = "xyxy", "voc"                              # VOC format: [x1, y1, x2, y2]
    # Conversion Codes
    XYWH2XYXY = "xywh_to_xyxy", "coco_to_voc"         # Convert from COCO to VOC
    XYWH2CXCYWHN = "xywh_to_cxcywhn", "coco_to_yolo"  # Convert from COCO to YOLO
    XYXY2XYWH = "xyxy_to_xywh", "voc_to_coco"         # Convert from VOC to COCO
    XYXY2CXCYWHN = "xyxy_to_cxcywhn", "voc_to_yolo"   # Convert from VOC to YOLO
    CXCYWHN2XYXY = "cxcywhn_to_xyxy", "yolo_to_voc"   # Convert from YOLO to VOC
    CXCYWHN2XYWH = "cxcywhn_to_xywh", "yolo_to_coco"  # Convert from YOLO to COCO

    @classmethod
    def formats(cls) -> list:
        """Return a list of standard bounding box formats."""
        return [
            cls.XYXY,
            cls.XYWH,
            cls.CXCYWHN,
        ]


class DeviceType(MultiStrEnum):
    """Enum for device types."""

    CPU = "cpu", "default"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class AlbumTargetType(MultiStrEnum):
    """Enum for albumentations target types."""

    IMAGE = "image"          # The primary input image(s) (e.g., (H, W, C)). Receives geometric, color, and intensity transforms. Uses standard interpolation for geometric transforms.
    BBOXES = "bboxes"        # Bounding boxes. Processed according to bbox_params. Requires bbox_params to be set.
    KEYPOINTS = "keypoints"  # Keypoints. Processed according to keypoint_params. Requires keypoint_params to be set.
    MASK = "mask"            # Segmentation mask(s) (e.g., (H, W)). Receives geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    MASKS = "masks"          # Multiple segmentation masks passed together (e.g., (N, H, W)). Processed like mask.
    MASK3D = "mask3d"        # A 3D mask (e.g., (D, H, W)). Receives 3D geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    MASKS3D = "masks3d"      # Multiple 3D masks (e.g., (N, D, H, W)). Processed like mask3d across the first dimension.
    VOLUME = "volume"        # A 3D volume (e.g., (D, H, W, C)). Receives 3D geometric transforms, and applicable 2D transforms slice-wise. Color/intensity transforms applied if treated as 'image'.
    VOLUMES = "volumes"      # Multiple 3D volumes (e.g., (N, D, H, W, C)). Processed like volume across the first dimension.


# --- Machine Learning ---

class RunMode(MultiStrEnum):
    """Enum for common ML run modes."""

    TRAIN = "train", "default"
    PREDICT = "predict"
    METRIC = "metric"


class Split(MultiStrEnum):
    """Enum for common dataset splits."""

    TRAIN = "train", "default"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


class Task(MultiStrEnum):
    """Enum for common ML tasks."""

    # --- Benchmark ---
    BENCHMARK = "benchmark"

    # --- Neural Networks ---
    BACKBONE = "backbone"

    # --- Computer Vision ---
    # -- Low-Level --
    # Restoration
    DEBLUR = "deblur"          # Deblurring
    DEHAZE = "dehaze"          # Dehazing
    DEMOIRE = "demoire"        # Demoireing
    DEMOSAIC = "demosaic"      # Demosaicing
    DENOISE = "denoise"        # Denoising
    DERAIN = "derain"          # Deraining
    DESNOW = "desnow"          # Desnowing
    INPAINT = "inpaint"        # Image Inpainting
    SUPER_RES = "super_res"    # Super-Resolution

    # Enhancement
    CC = "cc"                  # Color Correction
    CE = "ce"                  # Contrast Enhancement
    COLORIZATION = "colorization"
    LLIE = "llie"              # Low-Light Image Enhancement
    MEF = "mef"                # Multi-Exposure Fusion
    RETOUCH = "retouch"        # Image Retouching
    SHARPEN = "sharpen"        # Image Sharpening
    STYLE_TRANSFER = "style_transfer"
    TONE_MAPPING = "tone_mapping"

    # -- Mid-Level --
    # Keypoint
    POSE = "pose"              # Pose Estimation

    BGSUBTRACT = "bgsubtract"  # Background Subtraction
    MONODEPTH = "monodepth"    # Monocular-Depth Estimation
    OPTICAL_FLOW = "optical_flow"
    SEGMENT = "segment"        # Semantic Segmentation
    TRACK = "track"            # Tracking

    # -- High-Level --
    CLASSIFY = "classify"      # Classification
    DETECT = "detect"          # Object Detection

    # --- Generative AI ---
    IMG2IMG = "img2img"        # Image-to-Image Translation


class Precision(MultiStrEnum):
    """Enum for common numerical precisions."""

    FP32 = "fp32", "default"  # 32-bit floating point
    FP16 = "fp16"             # 16-bit floating point
    FP8 = "fp8"               # 8-bit floating point
    INT8 = "int8"             # 8-bit integer
    INT4 = "int4"             # 4-bit integer

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
