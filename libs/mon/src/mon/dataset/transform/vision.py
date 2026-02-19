#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vison Transform.

This module provides vision transformations.
"""

from __future__ import annotations

__all__ = [
    "NormalizeWithMask",
    "ResizeDivisibleBy",
]

from typing import Any, Callable, Literal, Self

import cv2
import numpy as np
from albucore import normalize, normalize_per_image
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    BasicTransform,
    DualTransform,
)
from albumentations.core.type_definitions import ALL_TARGETS, Targets
from numpy import ndarray
from pydantic import Field, model_validator

from mon.core import ALBUMENTATIONS, parse_imgsz


# ==============================================================================
# region PIXEL
# ==============================================================================

@ALBUMENTATIONS.register()
class NormalizeWithMask(BasicTransform):
    """Applies various normalization techniques to an image and masks. The
    specific normalization technique can be selected with the ``normalization``
    parameter.

    Standard normalization is applied using the formula:
        img = (img - mean * max_pixel_value) / (std * max_pixel_value).
        Other normalization techniques adjust the image based on global or per-channel statistics,
        or scale pixel values to a specified range.

    References:
        - ImageNet mean and std: https://pytorch.org/vision/stable/models.html
        - Inception preprocessing: https://keras.io/api/applications/inceptionv3/

    Examples:
        >>> import numpy as np
        >>> import albumentations as A
        >>> image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        >>> # Standard ImageNet normalization
        >>> transform = A.Normalize(
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225),
        ...     max_pixel_value=255.0,
        ...     p=1.0
        ... )
        >>> normalized_image = transform(image=image)["image"]
        >>>
        >>> # Min-max normalization
        >>> transform_minmax = A.Normalize(normalization="min_max", p=1.0)
        >>> normalized_image_minmax = transform_minmax(image=image)["image"]
    """

    _targets = (Targets.IMAGE, Targets.MASK, Targets.VOLUME)

    class InitSchema(BaseTransformInitSchema):
        mean: tuple[float, ...] | float | None
        std: tuple[float, ...] | float | None
        max_pixel_value: float | None
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ]

        @model_validator(mode="after")
        def _validate_normalization(self) -> Self:
            if (
                self.mean is None
                or self.std is None
                or (self.max_pixel_value is None and self.normalization == "standard")
            ):
                raise ValueError(
                    "mean, std, and max_pixel_value must be provided for "
                    "standard normalization.",
                )
            return self

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        mean: tuple[float, ...] | float | None = (0.485, 0.456, 0.406),
        std: tuple[float, ...] | float | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ] = "standard",
        p: float = 1.0,
    ):
        """Initializes the NormalizeWithMask transformation.

        Args:
            mean (tuple[float, ...] or float, optional): Mean values for
                standard normalization. Defaults to ImageNet mean values:
                (0.485, 0.456, 0.406).
            std (tuple[float, ...] or float, optional): Standard deviation
                values for standard normalization. Defaults to ImageNet
                standard deviation: (0.229, 0.224, 0.225).
            max_pixel_value (float, optional): Maximum possible pixel value,
                used for scaling in standard normalization. Defaults to 255.0.
            normalization (str, optional): Specifies the normalization technique
                to apply. Defaults to "standard".
                - "standard": Applies the formula
                    `(img - mean * max_pixel_value) / (std * max_pixel_value)`.
                    The default mean and std are based on ImageNet. You can use
                    mean and std values of (0.5, 0.5, 0.5) for inception
                    normalization. And mean values of (0, 0, 0) for std values
                    of (1, 1, 1) for YOLO.
                - "image": Normalizes the whole image based on its global mean
                    and standard deviation.
                - "image_per_channel": Normalizes the image per channel based
                    on each channel's mean and standard deviation.
                - "min_max": Scales the image pixel values to a [0, 1] range
                    based on the global minimum and maximum pixel values.
                - "min_max_per_channel": Scales each channel of the image pixel
                    values to a [0, 1] range based on the per-channel minimum
                    and maximum pixel values.
            p (float, optional): Probability of applying the transform.
                Defaults to 1.0.

        Note:
            - For "standard" normalization, ``mean``, ``std``, and
              ``max_pixel_value`` must be provided.
            - For other normalization types, these parameters are ignored.
            - For inception normalization, use mean values of (0.5, 0.5, 0.5).
            - For YOLO normalization, use mean values of (0, 0, 0) and std
              values of (1, 1, 1).
            - This transform is often used as a final step in image
              preprocessing pipelines to prepare images for neural network input.
        """
        super().__init__(p=p)
        self._mean = mean
        self._mean_np = np.array(mean, dtype=np.float32) * max_pixel_value
        self._std = std
        self._denominator = np.reciprocal(np.array(std, dtype=np.float32) * max_pixel_value)
        self._max_pixel_value = max_pixel_value
        self._normalization = normalization

    # --- Properties ---
    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Getter for the targets mapping.

        This property returns a dictionary that maps target types to their
        corresponding to apply methods.

        Returns:
            dict[str, Callable[..., Any]]: A dictionary mapping target types
                ("image", "mask", "volume") to their respective apply methods.
        """
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "volume": self.apply_to_volume,
            "volumes": self.apply_to_volumes,
        }

    # --- Apply ---
    def apply(self, img: ndarray, **params: Any) -> ndarray:
        """Applies normalization to the input image.

        Args:
            img (ndarray): Image to normalize.
            **params (Any): Additional parameters.

        Returns:
            numpy.ndarray: Normalized image.
        """
        if self._normalization == "standard":
            return normalize(img, self._mean_np, self._denominator)
        return normalize_per_image(img, self._normalization)

    def apply_to_images(self, images: ndarray, **params: Any) -> ndarray:
        """Applies normalization to a batch of images.

        Args:
            images (ndarray): Batch of images to normalize.
            **params (Any): Additional parameters.

        Returns:
            ndarray: Normalized batch of images.
        """
        return self.apply(images, **params)

    def apply_to_mask(self, mask: ndarray, **params: Any) -> ndarray:
        """Applies normalization to a mask.

        Args:
            mask (ndarray): Mask to normalize.
            **params (Any): Additional parameters.

        Returns:
            ndarray: Normalized mask.
        """
        return self.apply(mask, **params)

    def apply_to_masks(self, masks: ndarray, **params: Any) -> ndarray:
        """Applies normalization to a batch of masks.

        Args:
            masks (ndarray): Batch of masks to normalize.
            **params (Any): Additional parameters.

        Returns:
            ndarray: Normalized batch of masks.
        """
        return self.apply(masks, **params)

    def apply_to_volume(self, volume: ndarray, **params: Any) -> ndarray:
        """Applies normalization to a 3D volume.

        Args:
            volume (ndarray): 3D volume to normalize.
            **params (Any): Additional parameters.

        Returns:
            ndarray: Normalized volume.
        """
        return self.apply(volume, **params)

    def apply_to_volumes(self, volumes: ndarray, **params: Any) -> ndarray:
        """Applies normalization to a batch of 3D volumes.

        Args:
            volumes (ndarray): 3D volumes to normalize.
            **params (Any): Additional parameters.

        Returns:
            ndarray: Normalized batch of volumes.
        """
        return self.apply(volumes, **params)

# endregion


# ==============================================================================
# region PIXEL
# ==============================================================================

@ALBUMENTATIONS.register()
class ResizeDivisibleBy(DualTransform):
    """Resizes the input to a new size that is divisible by a given number."""

    _targets = ALL_TARGETS

    class InitSchema(BaseTransformInitSchema):
        height: int = Field(ge=0)
        width: int = Field(ge=0)
        divisor: int = Field(ge=1)
        area_for_downscale: Literal[None, "image", "image_mask"]
        interpolation: int
        mask_interpolation: int

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        height: int,
        width: int,
        divisor: int = 1,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        area_for_downscale: Literal[None, "image", "image_mask"] = None,
        p: float = 1,
    ):
        """Initializes the ResizeDivisibleBy transformation.

        Args:
            height (int): Desired height of the output image. If set to 0,
                the height will be determined based on the original image size.
            width (int): Desired width of the output image. If set to 0,
                the width will be determined based on the original image size.
            divisor (int, optional): The output dimensions will be made
                divisible by this number. Defaults to 1.
            interpolation (int, optional): Interpolation method for resizing
                images. Defaults to cv2.INTER_LINEAR.
            mask_interpolation (int, optional): Interpolation method for
                resizing masks. Defaults to cv2.INTER_NEAREST.
            area_for_downscale (str or None, optional): If set to "image", uses
                cv2.INTER_AREA for downscaling images. If set to "image_mask",
                uses cv2.INTER_AREA for both images and masks. If None, uses the
                specified interpolation methods. Defaults to None.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p=p)
        self._height = height
        self._width = width
        self._divisor = divisor
        self._interpolation = interpolation
        self._mask_interpolation = mask_interpolation
        self._area_for_downscale = area_for_downscale

    # --- Apply ---
    def apply(self, img: ndarray, **params: Any) -> ndarray:
        """Applies the resizing transformation to the input image.

        Args:
            img (ndarray): Input image to be resized.

        Returns:
            ndarray: Resized image.
        """
        h, w = img.shape[:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = parse_imgsz((new_h, new_w), divisor=self._divisor)

        is_downscale  = (new_h < h) or (new_w < w)
        interpolation = self._interpolation
        if self._area_for_downscale in ["image", "image_mask"] and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(img, (new_h, new_w), interpolation=interpolation)

    def apply_to_mask(self, mask: ndarray, **params: Any) -> ndarray:
        """Applies the resizing transformation to the input mask.

        Args:
            mask (ndarray): Input mask to be resized.

        Returns:
            ndarray: Resized mask.
        """
        h, w = mask.shape[:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = parse_imgsz((new_h, new_w), divisor=self._divisor)

        is_downscale  = (new_h < h) or (new_w < w)
        interpolation = self._mask_interpolation
        if self._area_for_downscale == "image_mask" and is_downscale:
            interpolation = cv2.INTER_AREA

        return fgeometric.resize(mask, (new_h, new_w), interpolation=interpolation)

    def apply_to_bboxes(self, bboxes: ndarray, **params: Any) -> ndarray:
        """Applies the resizing transformation to the input bounding boxes.

        Args:
            bboxes (ndarray): Input bounding boxes as an array of shape (N, 4+)
                in CXCYWHN format. The bounding boxes are scale invariant.

        Returns:
            ndarray: Resized bounding boxes.
        """
        # Bounding box coordinates are scale invariant so no need to adjust them
        return bboxes

    def apply_to_keypoints(self, keypoints: ndarray, **params: Any) -> ndarray:
        """Applies the resizing transformation to the input keypoints.

        Args:
            keypoints (ndarray): Input keypoints as an array of shape (N, 2+)
                where each keypoint is represented by (x, y, ...).

        Returns:
            ndarray: Resized keypoints.
        """
        h, w = params["shape"][:2]

        if self._height > 0 or self._width > 0:
            new_h, new_w = self._height, self._width
        else:
            new_h, new_w = h, w
        new_h, new_w = parse_imgsz((new_h, new_w), divisor=self._divisor)

        scale_x = self._width / new_w
        scale_y = self._height / new_h
        return fgeometric.keypoints_scale(keypoints, scale_x, scale_y)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
