#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for Albumentations augmentations.

This module provides base classes and mixins for Albumentations augmentations.
"""

from __future__ import annotations

__all__ = [
    "TARGET_TYPES",
]


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---

TARGET_TYPES = [
    "image",      # The primary input image(s) (e.g., [H, W, C]). Receives geometric, color, and intensity transforms. Uses standard interpolation for geometric transforms.
    "mask",       # Segmentation mask(s) (e.g., [H, W]). Receives geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks",      # Multiple segmentation masks passed together (e.g., [N, H, W]). Processed like mask.
    "bboxes",     # Bounding boxes. Processed according to bbox_params. Requires bbox_params to be set.
    "keypoints",  # Keypoints. Processed according to keypoint_params. Requires keypoint_params to be set.
    "volume",     # A 3D volume (e.g., [D, H, W, C]). Receives 3D geometric transforms, and applicable 2D transforms slice-wise. Color/intensity transforms applied if treated as 'image'.
    "volumes",    # Multiple 3D volumes (e.g., [N, D, H, W, C]). Processed like volume across the first dimension.
    "mask3d",     # A 3D mask (e.g., [D, H, W]). Receives 3D geometric transforms using nearest-neighbor interpolation. Does not receive color/intensity transforms.
    "masks3d"     # Multiple 3D masks (e.g., [N, D, H, W]). Processed like mask3d across the first dimension.
]


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
