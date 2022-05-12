#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Luv color space.

RGB to Luv color transformations were translated from scikit image's
rgb2luv and luv2rgb:
https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py
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
from one.imgproc.color.rgb import linear_rgb_to_rgb
from one.imgproc.color.rgb import rgb_to_bgr
from one.imgproc.color.rgb import rgb_to_linear_rgb
from one.imgproc.color.xyz import rgb_to_xyz
from one.imgproc.color.xyz import xyz_to_rgb
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "bgr_to_luv",
    "luv_to_bgr",
    "luv_to_rgb",
    "rgb_to_luv",
    "BgrToLuv",
    "LuvToBgr",
    "LuvToRgb",
    "RgbToLuv"
]


# MARK: - Functional

@dispatch(Tensor, float)
def bgr_to_luv(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert an BGR image to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to Luv.
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (Tensor[B, 3, H, W]):
            Luv version of the image.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_luv(rgb, eps)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, float)
def bgr_to_luv(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert an BGR image to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to Luv.
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (np.ndarray[B, 3, H, W]):
            Luv version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)


@dispatch(Tensor, float)
def luv_to_bgr(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert a Luv image to BGR.

    Args:
        image (Tensor[B, 3, H, W]):
            Luv image to be converted to BGR.
        eps (float):
            For numerically stability when dividing.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = luv_to_rgb(image, eps)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, float)
def luv_to_bgr(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert a Luv image to BGR.

    Args:
        image (Tensor[B, 3, H, W]):
            Luv image to be converted to BGR.
        eps (float):
            For numerically stability when dividing.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_LUV2BGR)


@dispatch(Tensor, float)
def luv_to_rgb(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert a Luv image to RGB.

    Args:
        image (Tensor[B, 3, H, W]):
            Luv image to be converted to RGB.
        eps (float):
            For numerically stability when dividing.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    L = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))

    a = u_w + u / (13 * L + eps)
    d = v_w + v / (13 * L + eps)
    c = 3 * y * (5 * d - 3)
    z = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x = -(c / (d + eps) + 3.0 * z)

    xyz_im  = torch.stack([x, y, z], -3)
    rgbs_im = xyz_to_rgb(xyz_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)
    
    return rgb_im


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, float)
def luv_to_rgb(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert a Luv image to RGB.

    Args:
        image (Tensor[B, 3, H, W]):
            Luv image to be converted to RGB.
        eps (float):
            For numerically stability when dividing.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_LUV2RGB)


@dispatch(Tensor, float)
def rgb_to_luv(image: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert an RGB image to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to Luv.
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (Tensor[B, 3, H, W]):
            Luv version of the image.
    """
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    
    xyz_im  = rgb_to_xyz(lin_rgb)
    x       = xyz_im[..., 0, :, :]
    y       = xyz_im[..., 1, :, :]
    z       = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L = torch.where(y > threshold,
                    116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0,
                    903.3 * y)

    # Compute reference white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))

    u_p = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p = (9 * y) / (x + 15 * y + 3 * z + eps)

    u   = 13 * L * (u_p - u_w)
    v   = 13 * L * (v_p - v_w)
    luv = torch.stack([L, u, v], dim=-3)

    return luv


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, float)
def rgb_to_luv(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert an RGB image to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to Luv.
        eps (float):
            For numerically stability when dividing.

    Returns:
        luv (np.ndarray[B, 3, H, W]):
            Luv version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_luv")
class BgrToLuv(nn.Module):
    """Convert an image from BGR to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_luv(image)


@TRANSFORMS.register(name="luv_to_bgr")
class LuvToBgr(nn.Module):
    """Convert an image from Luv to BGR.

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return luv_to_bgr(image)


@TRANSFORMS.register(name="luv_to_rgb")
class LuvToRgb(nn.Module):
    """Convert an image from Luv to RGB.

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return luv_to_rgb(image)


@TRANSFORMS.register(name="rgb_to_luv")
class RgbToLuv(nn.Module):
    """Convert an image from RGB to Luv. Image data is assumed to be in the
    range of [0.0, 1.0]. Luv color is computed using the D65 illuminant and
    Observer 2.

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] http://www.poynton.com/ColorFAQ.html
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_luv(image)
