#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements resizing augmentations."""

__all__ = [
    "NormalizeWithMask",
]

from typing import Any, Callable, Literal, Self

import numpy as np
from albucore import batch_transform, normalize, normalize_per_image
from albumentations.core.transforms_interface import (
    BaseTransformInitSchema,
    BasicTransform,
)
from albumentations.core.type_definitions import Targets
from pydantic import model_validator

from mon.core.factory import ALBUMENTATIONS


@ALBUMENTATIONS.register()
class NormalizeWithMask(BasicTransform):
    """Applies various normalization techniques to an image and masks. The
    specific normalization technique can be selected with the `normalization`
    parameter.

    Standard normalization is applied using the formula:
        `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`.
        Other normalization techniques adjust the image based on global or per-channel statistics,
        or scale pixel values to a specified range.

    Args:
        mean (tuple[float, float] | float | None): Mean values for standard normalization.
            For "standard" normalization, the default values are ImageNet mean values: (0.485, 0.456, 0.406).
        std (tuple[float, float] | float | None): Standard deviation values for standard normalization.
            For "standard" normalization, the default values are ImageNet standard deviation :(0.229, 0.224, 0.225).
        max_pixel_value (float | None): Maximum possible pixel value, used for scaling in standard normalization.
            Defaults to 255.0.
        normalization (Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"]):
            Specifies the normalization technique to apply. Defaults to "standard".
            - "standard": Applies the formula `(img - mean * max_pixel_value) / (std * max_pixel_value)`.
                The default mean and std are based on ImageNet. You can use mean and std values of (0.5, 0.5, 0.5)
                for inception normalization. And mean values of (0, 0, 0) and std values of (1, 1, 1) for YOLO.
            - "image": Normalizes the whole image based on its global mean and standard deviation.
            - "image_per_channel": Normalizes the image per channel based on each channel's mean and standard deviation.
            - "min_max": Scales the image pixel values to a [0, 1] range based on the global
                minimum and maximum pixel values.
            - "min_max_per_channel": Scales each channel of the image pixel values to a [0, 1]
                range based on the per-channel minimum and maximum pixel values.

        p (float): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image

    Image types:
        uint8, float32

    Note:
        - For "standard" normalization, `mean`, `std`, and `max_pixel_value` must be provided.
        - For other normalization types, these parameters are ignored.
        - For inception normalization, use mean values of (0.5, 0.5, 0.5).
        - For YOLO normalization, use mean values of (0, 0, 0) and std values of (1, 1, 1).
        - This transform is often used as a final step in image preprocessing pipelines to
          prepare images for neural network input.

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

    References:
        - ImageNet mean and std: https://pytorch.org/vision/stable/models.html
        - Inception preprocessing: https://keras.io/api/applications/inceptionv3/
    """
    
    _targets = (Targets.IMAGE, Targets.MASK, Targets.VOLUME)
    
    class InitSchema(BaseTransformInitSchema):
        mean: tuple[float, ...] | float | None
        std : tuple[float, ...] | float | None
        max_pixel_value: float | None
        normalization  : Literal[
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
                    "mean, std, and max_pixel_value must be provided for standard normalization.",
                )
            return self

    def __init__(
        self,
        mean: tuple[float, ...] | float | None = (0.485, 0.456, 0.406),
        std : tuple[float, ...] | float | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization  : Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ] = "standard",
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.mean        = mean
        self.mean_np     = np.array(mean, dtype=np.float32) * max_pixel_value
        self.std         = std
        self.denominator = np.reciprocal(
            np.array(std, dtype=np.float32) * max_pixel_value,
        )
        self.max_pixel_value = max_pixel_value
        self.normalization   = normalization
    
    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        """Get mapping of target keys to their corresponding processing functions for DualTransform.

        Returns:
            dict[str, Callable[..., Any]]: Dictionary mapping target keys to their processing functions.

        """
        return {
            "image"  : self.apply,
            "images" : self.apply_to_images,
            "mask"   : self.apply_to_mask,
            "masks"  : self.apply_to_masks,
            "volume" : self.apply_to_volume,
            "volumes": self.apply_to_volumes,
        }
    
    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to the input image.

        Args:
            img (np.ndarray): The input image to normalize.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The normalized image.

        """
        if self.normalization == "standard":
            return normalize(
                img,
                self.mean_np,
                self.denominator,
            )
        return normalize_per_image(img, self.normalization)
    
    def apply_to_images(self, images: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a batch of images.

        Args:
            images (np.ndarray): Batch of images to normalize with shape (batch, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized batch of images.

        """
        return self.apply(images, **params)
    
    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        """Apply resizing to the mask.

        Args:
            mask (np.ndarray): Mask to resize.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Resized mask.
        """
        return self.apply(mask, **params)
    
    def apply_to_masks(self, masks: np.ndarray, **params: Any) -> np.ndarray:
        """Apply transform to multiple masks.

        Args:
            masks (np.ndarray): Input masks as numpy array
            **params (Any): Additional parameters specific to the transform

        Returns:
            np.ndarray: Transformed masks as numpy array
        """
        return self.apply(masks, **params)
    
    def apply_to_volume(self, volume: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a 3D volume.

        Args:
            volume (np.ndarray): 3D volume to normalize with shape (depth, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized 3D volume.

        """
        return self.apply(volume, **params)
    
    def apply_to_volumes(self, volumes: np.ndarray, **params: Any) -> np.ndarray:
        """Apply normalization to a batch of 3D volumes.

        Args:
            volumes (np.ndarray): Batch of 3D volumes to normalize with shape (batch, depth, height, width, channels).
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Normalized batch of 3D volumes.

        """
        return self.apply(volumes, **params)
