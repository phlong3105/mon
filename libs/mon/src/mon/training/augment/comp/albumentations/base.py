#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for Albumentations augmentations.

This module provides base classes and mixins for Albumentations augmentations.
"""

from __future__ import annotations

__all__ = [
    "BasicTransform",
    "Compose",
    "TARGET_TYPES",
    "build_transforms",
]

from typing import Any

from albumentations.core.composition import Compose as Compose_

from mon.core import ALBUMENTATIONS
from .api import BasicTransform

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

class Compose(Compose_):
    """An extended version of ``albumentations.Compose`` that builds
    transformations from configuration dictionaries.
    """

    def __init__(self, transforms: list[Any], **kwargs):
        """Initialize a new instance.

        Args:
            transforms: List of transformations. If any element in ``transforms``
                is a dict, it will be used to build the corresponding
                transformation operation.
            **kwargs: Additional keyword arguments passed to the base
                ``albumentations.Compose``.
        """
        transforms = build_transforms(transforms)
        super().__init__(transforms, **kwargs)


def build_transforms(transforms: list[Any]) -> list[BasicTransform]:
    """Build a list of albumentations transformation operations.

    Args:
        transforms: A list of transformation operations. If any element in
            ``transforms`` is a dict, it will be used to build the corresponding
            transformation operation.

    Returns:
       A list of albumentations transformation operations.

    Raises:
        ValueError: If no valid transformation operations are found in ``transforms``.
    """
    transform_ops = []
    for i, t in enumerate(transforms):
        if isinstance(t, dict):
            t = ALBUMENTATIONS.build(**t)
        if t and isinstance(t, BasicTransform):
            transform_ops.append(t)

    if len(transform_ops) == 0:
        raise ValueError(f"``transforms`` must contain at least one valid transformation.")

    return transform_ops

# --- Mixins ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
