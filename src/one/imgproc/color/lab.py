#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Lab color space.

RGB to Lab color transformations were translated from scikit image's
rgb2lab and lab2rgb:
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
    "bgr_to_lab",
    "lab_to_bgr",
    "lab_to_rgb",
    "rgb_to_lab",
    "BgrToLab",
    "LabToBgr",
    "LabToRgb",
    "RgbToLab"
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_lab(image: Tensor) -> Tensor:
    """Convert BGR image to Lab. Image data is assumed to be in the range
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to Lab.

    Returns:
        lab (Tensor[B, 3, H, W]):
            Lab version of the image. L channel values are in the range
            [0, 100]. a and b are in the range [-127, 127].
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_lab(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to Lab. Image data is assumed to be in the range
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to Lab.

    Returns:
        lab (np.ndarray[B, 3, H, W]):
            Lab version of the image. L channel values are in the range
            [0, 100]. a and b are in the range [-127, 127].
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


@dispatch(Tensor, bool)
def lab_to_bgr(image: Tensor, clip: bool = True) -> Tensor:
    """Convert a Lab image to BGR.

    Args:
        image (Tensor[B, 3, H, W]):
            Lab image to be converted to BGR.
        clip (bool):
            Whether to apply clipping to insure output BGR values in range
            [0.0 1.0].

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = lab_to_rgb(image)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, bool)
def lab_to_bgr(image: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert a Lab image to RGB.

    Args:
        image (np.ndarray[B, 3, H, W]):
            Lab image to be converted to bgr.
        clip (bool):
            Whether to apply clipping to insure output bgr values in range
            [0.0 1.0].

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


@dispatch(Tensor, bool)
def lab_to_rgb(image: Tensor, clip: bool = True) -> Tensor:
    """Convert a Lab image to RGB.

    Args:
        image (Tensor[B, 3, H, W]):
            Lab image to be converted to RGB.
        clip (bool):
            Whether to apply clipping to insure output RGB values in range
            [0.0 1.0].

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    L  = image[..., 0, :, :]
    a  = image[..., 1, :, :]
    _b = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # If color data out of range: Z < 0
    fz   = fz.clamp(min=0.0)
    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz   = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    )[..., :, None, None]
    xyz_im  = xyz * xyz_ref_white
    rgbs_im = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to [0.0, 1.0] https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, bool)
def lab_to_rgb(image: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert a Lab image to RGB.

    Args:
        image (np.ndarray[B, 3, H, W]):
            Lab image to be converted to RGB.
        clip (bool):
            Whether to apply clipping to insure output RGB values in range
            [0.0 1.0].

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)


@dispatch(Tensor)
def rgb_to_lab(image: Tensor) -> Tensor:
    """Convert RGB image to Lab. Image data is assumed to be in the range
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to Lab.

    Returns:
        lab (Tensor[B, 3, H, W]):
            Lab version of the image. L channel values are in the range
            [0, 100]. a and b are in the range [-127, 127].
    """
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    xyz_im  = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white  = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power     = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale     = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int   = torch.where(xyz_normalized > threshold, power, scale)

    x = xyz_int[..., 0, :, :]
    y = xyz_int[..., 1, :, :]
    z = xyz_int[..., 2, :, :]

    L  = (116.0 * y) - 16.0
    a  = 500.0 * (x - y)
    _b = 200.0 * (y - z)

    lab = torch.stack([L, a, _b], dim=-3)
    return lab


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to Lab. Image data is assumed to be in the range
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to Lab.

    Returns:
        lab (np.ndarray[B, 3, H, W]):
            Lab version of the image. L channel values are in the range
            [0, 100]. a and b are in the range [-127, 127].
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_lab")
class BgrToLab(nn.Module):
    """Convert an image from BGR to Lab. Image data is assumed to be in the
    range of [0.0 1.0]. Lab color is computed using the D65 illuminant and
    Observer 2.
 
    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_lab(image)


@TRANSFORMS.register(name="lab_to_bgr")
class LabToBgr(nn.Module):
    """Convert an image from Lab to BGR.

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray, clip: bool = True) -> TensorOrArray:
        return lab_to_bgr(image, clip)


@TRANSFORMS.register(name="lab_to_rgb")
class LabToRgb(nn.Module):
    """Convert an image from Lab to RGB.
 
    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray, clip: bool = True) -> TensorOrArray:
        return lab_to_rgb(image, clip)


@TRANSFORMS.register(name="rgb_to_lab")
class RgbToLab(nn.Module):
    """Convert an image from RGB to Lab. Image data is assumed to be in the
    range of [0.0 1.0]. Lab color is computed using the D65 illuminant and
    Observer 2.

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html
        [2] https://www.easyrgb.com/en/math.php
        [3] https://github.com/torch/image/blob
        /dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_lab(image)
