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
    "DefaultEnumMeta",
    "DeviceType",
    "ImageExtension",
    "MemoryUnit",
    "Precision",
    "RunMode",
    "Split",
    "StrEnum",
    "Task",
    "VideoExtension",
    "WeightExtension",
]

from .base import DefaultEnumMeta, StrEnum


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Units ---

class MemoryUnit(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for memory units."""

    B = "B"
    KB = "KB"
    MB = "MB"
    GB = "GB"
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

class ConfigExtension(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for configuration file extensions."""

    CFG = ".cfg"
    CONFIG = ".config"
    JSON = ".json"
    TXT = ".txt"
    YAML = ".yaml"
    YML = ".yml"
    DEFAULT = YAML


class ImageExtension(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common image file extensions."""

    ARW = ".arw"
    BMP = ".bmp"
    DNG = ".dng"
    JPEG = ".jpeg"
    JPG = ".jpg"
    PNG = ".png"
    PPM = ".ppm"
    RAF = ".raf"
    TIF = ".tif"
    TIFF = ".tiff"
    DEFAULT = JPG


class VideoExtension(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common video file extensions."""

    AVI = ".avi"
    M4V = ".m4v"
    MKV = ".mkv"
    MOV = ".mov"
    MP4 = ".mp4"
    MPEG = ".mpeg"
    MPG = ".mpg"
    WMV = ".wmv"
    DEFAULT = MP4


class WeightExtension(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common model weight file extensions."""

    CKPT = ".ckpt"
    ONNX = ".onnx"
    PT = ".pt"
    PTH = ".pth"
    TAR = ".tar"
    WEIGHTS = ".weights"
    DEFAULT = PT


# --- Data ---

class BBoxFormat(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for bounding box formats and conversion codes."""

    # Formats
    XYWH = "xywh"  # COCO format: [x, y, w, h]
    XYXY = "xyxy"  # VOC format: [x1, y1, x2, y2]
    CXCYWHN = "cxcywhn"  # YOLO format: [cx, cy, w, h] normalized
    # Conversion Codes
    XYWH2XYXY = "xywh_to_xyxy"  # Convert from COCO to VOC
    XYWH2CXCYWHN = "xywh_to_cxcywhn"  # Convert from COCO to YOLO
    XYXY2XYWH = "xyxy_to_xywh"  # Convert from VOC to COCO
    XYXY2CXCYWHN = "xyxy_to_cxcywhn"  # Convert from VOC to YOLO
    CXCYWHN2XYXY = "cxcywhn_to_xyxy"  # Convert from YOLO to VOC
    CXCYWHN2XYWH = "cxcywhn_to_xywh"  # Convert from YOLO to COCO
    DEFAULT = CXCYWHN

    @classmethod
    def formats(cls) -> list:
        """Return a list of standard bounding box formats."""
        return [
            cls.XYXY,
            cls.XYWH,
            cls.CXCYWHN,
        ]


class DeviceType(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"
    DEFAULT = CPU


class AlbumTargetType(StrEnum, metaclass=DefaultEnumMeta):
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
    DEFAULT = IMAGE


# --- Machine Learning ---

class RunMode(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common ML run modes."""

    TRAIN = "train"
    PREDICT = "predict"
    METRIC = "metric"
    DEFAULT = PREDICT


class Split(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"
    DEFAULT = TRAIN


class Task(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common ML tasks."""

    # --- Benchmark ---
    BENCHMARK = "benchmark"

    # --- Neural Networks ---
    BACKBONE = "backbone"

    # --- Generative AI ---
    IMG2IMG = "img2img"  # Image-to-Image Translation

    # --- Computer Vision ---
    BGSUBTRACT = "bgsubtract"  # Background Subtraction
    CLASSIFY = "classify"  # Classification
    DETECT = "detect"  # Object Detection
    ENHANCE = "enhance"  # Image Enhancement
    MONODEPTH = "monodepth"  # Monocular-Depth Estimation
    POSE = "pose"  # Pose Estimation
    RESTORE = "restore"  # Image Restoration
    SEGMENT = "segment"  # Semantic Segmentation
    TRACK = "track"  # Tracking


class Precision(StrEnum, metaclass=DefaultEnumMeta):
    """Enum for common numerical precisions."""

    FP32 = "fp32"  # 32-bit floating point
    FP16 = "fp16"  # 16-bit floating point
    FP8 = "fp8"  # 8-bit floating point
    INT8 = "int8"  # 8-bit integer
    INT4 = "int4"  # 4-bit integer
    DEFAULT = FP32

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
