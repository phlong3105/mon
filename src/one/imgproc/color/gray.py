#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Grayscale color space.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
from torch import Tensor

from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.color.rgb import bgr_to_rgb
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "bgr_to_grayscale",
    "grayscale_to_rgb",
    "rgb_to_grayscale",
    "BgrToGrayscale",
    "GrayscaleToRgb",
    "RgbToGrayscale",
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_grayscale(image: Tensor) -> Tensor:
    """Convert a BGR image to grayscale. Image data is assumed to be in the
    range of [0.0, 1.0]. First flips to RGB, then converts.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR image to be converted to grayscale.

    Returns:
        grayscale (TensorTensor[B, 1, H, W]):
            Grayscale version of the image.
    """
    image_rgb = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. Image data is assumed to be in the
    range of [0.0, 1.0]. First flips to RGB, then converts.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR image to be converted to grayscale.

    Returns:
        grayscale (np.ndarray[B, 1, H, W]):
            Grayscale version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@dispatch(Tensor)
def grayscale_to_rgb(image: Tensor) -> Tensor:
    """Convert a grayscale image to RGB version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (Tensor[B, 1, H, W]):
            Grayscale image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    rgb = torch.cat([image, image, image], dim=-3)

    # NOTE: we should find a better way to raise this kind of warnings
    # if not torch.is_floating_point(image):
    #     warnings.warn(f"Input image is not of float dtype. Got: {image.dtype}")

    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 1, H, W]):
            Grayscale image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


@dispatch(Tensor, list)
def rgb_to_grayscale(
    image: Tensor, rgb_weights: list[float] = [0.299, 0.587, 0.114]
) -> Tensor:
    """Convert an RGB image to grayscale version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            RGB image to be converted to grayscale.
        rgb_weights (list[float]):
            Weights that will be applied on each channel (RGB). Sum of the
            weights should add up to one.
    
    Returns:
        grayscale (Tensor[B, 1, H, W]):
            Grayscale version of the image.
    """
    rgb_weights = torch.FloatTensor(rgb_weights)
    if not isinstance(rgb_weights, Tensor):
        raise TypeError(f"`rgb_weights` must be a `Tensor`. "
                        f"But got: {type(rgb_weights)}.")
    if rgb_weights.shape[-1] != 3:
        raise ValueError(f"`rgb_weights` must have a shape of [*, 3]. "
                         f"But got: {rgb_weights.shape}.")
    
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise ValueError(f"`image` and `rgb_weights` must have the same dtype. "
                         f"But got: {image.dtype} and {rgb_weights.dtype}.")

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b


@dispatch(np.ndarray, list)
def rgb_to_grayscale(
    image: np.ndarray, rgb_weights: list[float] = [0.299, 0.587, 0.114]
) -> np.ndarray:
    """Convert an RGB image to grayscale version of image. Image data is
    assumed to be in the range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB image to be converted to grayscale.
        rgb_weights (list[float]):
            Weights that will be applied on each channel (RGB). Sum of the
            weights should add up to one.

    Returns:
        grayscale (np.ndarray[B, 1, H, W]):
            Grayscale version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_grayscale")
class BgrToGrayscale(nn.Module):
    """Module to convert a BGR image to grayscale version of image. Image
    data is assumed to be in the range of [0.0, 1.0]. First flips to RGB, then
    converts.

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_grayscale(image)


@TRANSFORMS.register(name="grayscale_to_rgb")
class GrayscaleToRgb(nn.Module):
    """Module to convert a grayscale image to RGB version of image. Image
    data is assumed to be in the range of [0.0, 1.0].

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return grayscale_to_rgb(image)


@TRANSFORMS.register(name="rgb_to_grayscale")
class RgbToGrayscale(nn.Module):
    """Module to convert a RGB image to grayscale version of image. Image
    data is assumed to be in the range of [0.0, 1.0].

    Reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    # MARK: Magic Functions
    
    def __init__(self, rgb_weights: list[float] = [0.299, 0.587, 0.114]):
        super().__init__()
        self.rgb_weights = rgb_weights

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)
