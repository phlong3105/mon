#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for resizing transformation.

This module provides a custom Albumentations transformation that resizes input
images, masks, bounding boxes, and keypoints to a new size that is divisible by
a specified number. It supports different interpolation methods for images and
masks, and can automatically select the appropriate interpolation method for
downscaling.
"""

from __future__ import annotations

__all__ = [
    "ResizeDivisibleBy",
]

from typing import Any, Literal

import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS
from pydantic import Field

from mon.core import ALBUMENTATIONS, image as I


@ALBUMENTATIONS.register()
class ResizeDivisibleBy(DualTransform):
    """Resizes the input to a new size that is divisible by a given number.

    Attributes:
        _targets (List[str]): List of target types that the transformation
            can be applied to. Supports image, mask, bboxes, keypoints, volume,
            and mask3d.
    """

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        height : int = Field(ge=0)
        width  : int = Field(ge=0)
        divisor: int = Field(ge=1)
        area_for_downscale: Literal[None, "image", "image_mask"]
        interpolation     : Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        height : int,
        width  : int,
        divisor: int = 1,
        interpolation     : Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_LINEAR,
        mask_interpolation: Literal[
            cv2.INTER_NEAREST,
            cv2.INTER_NEAREST_EXACT,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_LINEAR_EXACT,
        ] = cv2.INTER_NEAREST,
        area_for_downscale: Literal[None, "image", "image_mask"] = None,
        p: float = 1,
    ):
        """Initializes the ResizeDivisibleBy transformation.

        Args:
            height (int): Desired height of the output image. If set to 0,
                the height will be determined based on the original image size.
            width (int): Desired width of the output image. If set to 0,
                the width will be determined based on the original image size.
            divisor (int): The output dimensions will be made divisible by this
                number. Defaults to 1.
            interpolation (int): Interpolation method for resizing images.
                Defaults to cv2.INTER_LINEAR.
            mask_interpolation (int): Interpolation method for resizing masks.
                Defaults to cv2.INTER_NEAREST.
            area_for_downscale (str or None): If set to "image", uses
                cv2.INTER_AREA for downscaling images. If set to "image_mask",
                uses cv2.INTER_AREA for both images and masks. If None, uses the
                specified interpolation methods. Defaults to None.
            p (float): Probability of applying the transformation. Defaults to 1.0.
        """
        super().__init__(p=p)
        self._height             = height
        self._width              = width
        self._divisor            = divisor
        self._interpolation      = interpolation
        self._mask_interpolation = mask_interpolation
        self._area_for_downscale = area_for_downscale

    # --- Apply ---
    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Applies the resizing transformation to the input image.

        Args:
            img (numpy.ndarray): Input image to be resized.

        Returns:
            numpy.ndarray: Resized image.
        """
        h, w = img.shape[:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = I.imgsz((new_h, new_w), divisor=self._divisor)

        is_downscale  = (new_h < h) or (new_w < w)
        interpolation = self._interpolation
        if self._area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(img, (new_h, new_w), interpolation=interpolation)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Applies the resizing transformation to the input mask.

        Args:
            mask (numpy.ndarray): Input mask to be resized.

        Returns:
            numpy.ndarray: Resized mask.
        """
        h, w = mask.shape[:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = I.imgsz((new_h, new_w), divisor=self._divisor)

        is_downscale  = (new_h < h) or (new_w < w)
        interpolation = self._mask_interpolation
        if self._area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (new_h, new_w), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        """Applies the resizing transformation to the input bounding boxes.

        Args:
            bboxes (numpy.ndarray): Input bounding boxes as a numpy array of shape
                (N, 4+) in CXCYWHN format. The bounding boxes are scale invariant.

        Returns:
            numpy.ndarray: Resized bounding boxes.
        """
        # Bounding box coordinates are scale invariant so no need to adjust them
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        """Applies the resizing transformation to the input keypoints.

        Args:
            keypoints (numpy.ndarray): Input keypoints as a numpy array of shape
                (N, 2+) where each keypoint is represented by (x, y, ...).

        Returns:
            numpy.ndarray: Resized keypoints.
        """
        h, w = params["shape"][:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = I.imgsz((new_h, new_w), divisor=self._divisor)

        scale_x = self._width / new_w
        scale_y = self._height / new_h
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)
