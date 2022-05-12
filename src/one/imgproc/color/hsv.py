#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HSV color space.
"""

from __future__ import annotations

import math

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
    "bgr_to_hsv",
    "hsv_to_bgr",
    "hsv_to_rgb",
    "rgb_to_hsv",
    "BgrToHsv",
    "HsvToBgr",
    "HsvToRgb",
    "RgbToHsv"
]


# MARK: - Functional

@dispatch(Tensor, float)
def bgr_to_hsv(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert an image from BGR to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to HSV.
        eps (float):
            Scalar to enforce numarical stability.

    Returns:
        hsv (Tensor[B, 3, H, W]):
            HSV version of the image. H channel values are in the range
            [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_hsv(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert an image from HSV to BGR. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to HSV.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            HSV version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


@dispatch(Tensor)
def hsv_to_bgr(image: Tensor) -> Tensor:
    """Convert an image from HSV to BGR. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            HSV Image to be converted to HSV.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = hsv_to_rgb(image)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def hsv_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from HSV to BGR. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            HSV Image to be converted to HSV.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


@dispatch(Tensor)
def hsv_to_rgb(image: Tensor) -> Tensor:
    """Convert an image from HSV to RGB. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            HSV Image to be converted to HSV.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    h   = image[..., 0, :, :] / (2 * math.pi)
    s   = image[..., 1, :, :]
    v   = image[..., 2, :, :]

    hi  = torch.floor(h * 6) % 6
    f   = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p   = v * (one - s)
    q   = v * (one - f * s)
    t   = v * (one - (one - f) * s)

    hi      = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    
    out = torch.stack((
        v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q
    ), dim=-3)
    out = torch.gather(out, -3, indices)
    return out


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def hsv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from HSV to RGB. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            HSV Image to be converted to HSV.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


@dispatch(Tensor, float)
def rgb_to_hsv(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert an image from RGB to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to HSV.
        eps (float):
            Scalar to enforce numarical stability.

    Returns:
        hsv (Tensor[B, 3, H, W]):
            HSV version of the image. H channel values are in the range
            [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac              = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac     = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h   = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h   = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h   = (h / 6.0) % 1.0
    h  *= 2.0 * math.pi  # We return 0/2pi output
    hsv = torch.stack((h, s, v), dim=-3)
    
    return hsv


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert an image from HSV to RGB. FH channel values are assumed to be
    in the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to HSV.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            HSV version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_hsv")
class BgrToHsv(nn.Module):
    """Convert an image from BGR to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        eps (float):
            Scalar to enforce numarical stability. Default: `1e-8`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_hsv(image, self.eps)


@TRANSFORMS.register(name="hsv_to_bgr")
class HsvToBgr(nn.Module):
    """Convert an image from HSV to BGR. H channel values are assumed to be in
    the range [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return hsv_to_bgr(image)


@TRANSFORMS.register(name="hsv_to_rgb")
class HsvToRgb(nn.Module):
    """Convert an image from HSV to RGB. H channel values are assumed to be in
    the range [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return hsv_to_rgb(image)


@TRANSFORMS.register(name="rgb_to_hsv")
class RgbToHsv(nn.Module):
    """Convert an image from RGB to HSV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        eps (float):
            Scalar to enforce numarical stability. Default: `1e-8`.
    """

    # MARK: Magic Functions
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_hsv(image, self.eps)
