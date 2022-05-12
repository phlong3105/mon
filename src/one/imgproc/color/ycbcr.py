#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ycbcr color space.
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
    "bgr_to_ycrcb",
    "rgb_to_ycrcb",
    "ycrcb_to_rgb",
    "ycrcb_to_bgr",
    "BgrToYcrcb",
    "RgbToYcrcb",
    "YcrcbToBgr",
    "YcrcbToRgb",
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_ycrcb(image: Tensor) -> Tensor:
    """Convert an RGB image to YCrCb.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to YCrCb.

    Returns:
        ycrcb (Tensor[B, 3, H, W]):
            YCrCb version of the image.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_ycrcb(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_ycrcb(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YCbCr.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to YCbCr.

    Returns:
        ycbcr (np.ndarray[B, 3, H, W]):
            YCbCr version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


@dispatch(Tensor)
def rgb_to_ycrcb(image: Tensor) -> Tensor:
    """Convert an RGB image to YCrCb.

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to YCrCb.

    Returns:
        ycrcb (Tensor[B, 3, H, W]):
            YCrCb version of the image.
    """
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = 0.5
    y     = 0.299 * r + 0.587 * g + 0.114 * b
    cb    = (b - y) * 0.564 + delta
    cr    = (r - y) * 0.713 + delta
    ycrcb = torch.stack([y, cr, cb], -3)
    
    return ycrcb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgb_to_ycrcb(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YCbCr.

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to YCbCr.

    Returns:
        ycbcr (np.ndarray[B, 3, H, W]):
            YCbCr version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)


@dispatch(Tensor)
def ycrcb_to_bgr(image: Tensor) -> Tensor:
    """Convert an YCrCb image to BGR. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            YCrCb Image to be converted to RGB.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = ycrcb_to_rgb(image)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def ycrcb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an YCrCb image to BGR. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            YCrCb Image to be converted to RGB.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)


@dispatch(Tensor)
def ycrcb_to_rgb(image: Tensor) -> Tensor:
    """Convert an YCrCb image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            YCrCb Image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    y  = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta      = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r   = y + 1.403 * cr_shifted
    g   = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b   = y + 1.773 * cb_shifted
    rgb = torch.stack([r, g, b], -3)

    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def ycrcb_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an YCrCb image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            YCrCb Image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_ycrcb")
class BgrToYcrcb(nn.Module):
    """Convert an image from BGR to YCbCr. Image data is assumed to be in
    the range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_ycrcb(image)
    

@TRANSFORMS.register(name="rgb_to_ycrcb")
class RgbToYcrcb(nn.Module):
    """Convert an image from RGB to YCbCr. Image data is assumed to be in
    the range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_ycrcb(image)


@TRANSFORMS.register(name="ycrcb_to_bgr")
class YcrcbToBgr(nn.Module):
    """Convert an image from YCbCr to BGR. Image data is assumed to be in
    the range of [0.0, 1.0].
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return ycrcb_to_bgr(image)


@TRANSFORMS.register(name="ycrcb_to_rgb")
class YcrcbToRgb(nn.Module):
    """Convert an image from YCbCr to RGB. Image data is assumed to be in
    the range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return ycrcb_to_rgb(image)
