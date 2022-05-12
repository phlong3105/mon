#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""XYZ color space.
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
from one.imgproc.color.rgb import rgb_to_bgr
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "bgr_to_xyz",
    "rgb_to_xyz",
    "xyz_to_bgr",
    "xyz_to_rgb",
    "BgrToXyz",
    "RgbToXyz",
    "XyzToRgb",
    "XyzToRgb"
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_xyz(image: Tensor) -> Tensor:
    """Convert an BGR image to XYZ.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to XYZ.

    Returns:
        xyz (Tensor[B, 3, H, W]):
            XYZ version of the image.
    """
    rgb = bgr_to_rgb(image)
    return bgr_to_xyz(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_xyz(image: np.ndarray) -> np.ndarray:
    """Convert an BGR image to XYZ.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to XYZ.

    Returns:
        xyz (np.ndarray[B, 3, H, W]):
            XYZ version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)


@dispatch(Tensor)
def rgb_to_xyz(image: Tensor) -> Tensor:
    """Convert an RGB image to XYZ.

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to XYZ.

    Returns:
        xyz (Tensor[B, 3, H, W]):
            XYZ version of the image.
    """
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]

    x   = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y   = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z   = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz = torch.stack([x, y, z], -3)
    
    return xyz


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgb_to_xyz(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to XYZ.

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to XYZ.

    Returns:
        xyz (np.ndarray[B, 3, H, W]):
            XYZ version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)


@dispatch(Tensor)
def xyz_to_bgr(image: Tensor) -> Tensor:
    """Convert a XYZ image to BGR.

    Args:
        image (Tensor[B, 3, H, W]):
            XYZ Image to be converted to RGB.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = xyz_to_rgb(image)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def xyz_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a XYZ image to BGR.

    Args:
        image (np.ndarray[B, 3, H, W]):
            XYZ Image to be converted to BGR.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_XYZ2BGR)


@dispatch(Tensor)
def xyz_to_rgb(image: Tensor) -> Tensor:
    """Convert a XYZ image to RGB.

    Args:
        image (Tensor[B, 3, H, W]):
            XYZ Image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]

    r = ( 3.2404813432005266 * x +
         -1.5371515162713185 * y +
         -0.4985363261688878 * z)
    g = (-0.9692549499965682 * x +
          1.8759900014898907 * y +
          0.0415559265582928 * z)
    b = ( 0.0556466391351772 * x +
         -0.2040413383665112 * y +
          1.0573110696453443 * z)
    rgb = torch.stack([r, g, b], dim=-3)
    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def xyz_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a XYZ image to RGB.

    Args:
        image (np.ndarray[B, 3, H, W]):
            XYZ Image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_XYZ2RGB)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_xyz")
class BgrToXyz(nn.Module):
    """Convert an image from BGR to XYZ. Image data is assumed to be in the
    range of [0.0, 1.0].

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_xyz(image)


@TRANSFORMS.register(name="rgb_to_xyz")
class RgbToXyz(nn.Module):
    """Convert an image from RGB to XYZ. Image data is assumed to be in the
    range of [0.0, 1.0].
 
    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_xyz(image)


@TRANSFORMS.register(name="xyz_to_bgr")
class XyzToRgb(nn.Module):
    """Converts an image from XYZ to BGR.

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return xyz_to_bgr(image)


@TRANSFORMS.register(name="xyz_to_rgb")
class XyzToRgb(nn.Module):
    """Converts an image from XYZ to RGB.
 
    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return xyz_to_rgb(image)
