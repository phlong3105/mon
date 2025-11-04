#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements resizing augmentations."""

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

from mon.core.dtypes import image as I
from mon.core.factory import ALBUMENTATIONS


@ALBUMENTATIONS.register()
class ResizeDivisibleBy(DualTransform):
    """Resize the input to a new size that is divisible by a given number.

    Args:
        height: Desired height of the output.
        width: Desired width of the output.
        divisor: A number by which the output size should be divisible. Default: ``1``.
        interpolation: Flag that is used to specify the interpolation algorithm.
            Should be one of: ``cv2.INTER_NEAREST``, ``cv2.INTER_LINEAR``,
            ``cv2.INTER_CUBIC``, ``cv2.INTER_AREA``, ``cv2.INTER_LANCZOS4``.
            Default: ``cv2.INTER_LINEAR``.
        mask_interpolation: Flag that is used to specify the interpolation
            algorithm for mask. Should be one of: ``cv2.INTER_NEAREST``,
            ``cv2.INTER_LINEAR``, ``cv2.INTER_CUBIC``, ``cv2.INTER_AREA``,
            ``cv2.INTER_LANCZOS4``. Default: ``cv2.INTER_NEAREST``.
        area_for_downscale: Controls automatic use of ``INTER_AREA`` interpolation
            for downscaling. Options:
                - ``None``: No automatic interpolation selection, always use
                    the specified interpolation method
                - ``"image"``: Use ``INTER_AREA`` when downscaling images,
                    retain specified interpolation for upscaling and masks
                - ``"image_mask"``: Use ``INTER_AREA`` when downscaling both
                    images and masks
            Default: ``None``.
        p: probability of applying the transform. Default: ``1.0``.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        ``'uint8'``, ``'float32'``.
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
        super().__init__(p=p)
        self.height             = height
        self.width              = width
        self.divisor            = divisor
        self.interpolation      = interpolation
        self.mask_interpolation = mask_interpolation
        self.area_for_downscale = area_for_downscale
        
    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        height, width = img.shape[:2]
        
        if self.height > 0 or self.width > 0:
            new_height, new_width = self.height, self.width
        else:
            new_height, new_width = height, width
        new_height, new_width = I.imgsz((new_height, new_width), divisor=self.divisor)
        
        is_downscale  = (new_height < height) or (new_width < width)
        interpolation = self.interpolation
        if self.area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(img, (new_height, new_width), interpolation=interpolation)

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        height, width = mask.shape[:2]
        
        if self.height > 0 or self.width > 0:
            new_height, new_width = self.height, self.width
        else:
            new_height, new_width = height, width
        new_height, new_width = I.imgsz((new_height, new_width), divisor=self.divisor)
        
        is_downscale  = (new_height < height) or (new_width < width)
        interpolation = self.mask_interpolation
        if self.area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (new_height, new_width), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, **params: Any) -> np.ndarray:
        # Bounding box coordinates are scale invariant
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, **params: Any) -> np.ndarray:
        height, width = params["shape"][:2]
        
        if self.height > 0 or self.width > 0:
            new_height, new_width = self.height, self.width
        else:
            new_height, new_width = height, width
        new_height, new_width = I.imgsz((new_height, new_width), divisor=self.divisor)
        
        scale_x = self.width  / new_width
        scale_y = self.height / new_height
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)
