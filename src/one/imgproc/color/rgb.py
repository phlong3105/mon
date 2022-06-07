#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RGB color space.
"""

from __future__ import annotations

from typing import cast
from typing import Union

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
from torch import Tensor

from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "bgr_to_rgb",
    "bgr_to_rgba",
    "linear_rgb_to_rgb",
    "rgb_to_bgr",
    "rgb_to_linear_rgb",
    "rgb_to_rgba",
    "rgba_to_bgr",
    "rgba_to_rgb",
    "BgrToRgb",
    "BgrToRgba",
    "LinearRgbToRgb",
    "RgbaToBgr",
    "RgbaToRgb",
    "RgbToBgr",
    "RgbToLinearRgb",
    "RgbToRgba"
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_rgb(image: Tensor) -> Tensor:
    """Convert a BGR image to RGB.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to BGR.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    # Flip image channels
    rgb = image.flip(-3)
    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to BGR.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dispatch(Tensor, (float, Tensor))
def bgr_to_rgba(image: Tensor, alpha_val: Union[float, Tensor]) -> Tensor:
    """Convert an image from BGR to RGBA.

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to RGBA.
        alpha_val (float, Tensor[B, 1, H, W]):
            A float number or tensor for the alpha value.

    Returns:
        rgba (Tensor[B, 4, H, W]):
            RGBA version of the image.

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(f"`alpha_val` must be a `float` or `Tensor`. "
                        f"But got: {type(alpha_val)}.")
  
    # Convert first to RGB, then add alpha channel
    rgb  = bgr_to_rgb(image)
    rgba = rgb_to_rgba(rgb, alpha_val)

    return rgba


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, (float, np.ndarray))
def bgr_to_rgba(image: np.ndarray, alpha_val: Union[float, np.ndarray]) -> np.ndarray:
    """Convert an image from BGR to RGBA.

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to RGBA.
        alpha_val (float, np.ndarray[B, 1, H, W]):
            A float number or tensor for the alpha value.

    Returns:
        rgba (np.ndarray[B, 4, H, W]):
            RGBA version of the image.

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)


@dispatch(Tensor)
def linear_rgb_to_rgb(image: Tensor) -> Tensor:
    """Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image (Tensor[B, 3, H, W]):
            Linear RGB Image to be converted to sRGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            sRGB version of the image.
    """
    threshold = 0.0031308
    rgb       = torch.where(
        image > threshold,
        1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055,
        12.92 * image
    )
    return rgb


def rgb_to_bgr(image: TensorOrArray) -> TensorOrArray:
    """Convert an RGB image to BGR.

    Args:
        image (TensorOrArray[B, 3, H, W]):
            RGB Image to be converted to BGR.

    Returns:
        bgr (TensorOrArray[B, 3, H, W]):
            BGR version of the image.
    """
    return bgr_to_rgb(image)


@dispatch(Tensor)
def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    """Convert an sRGB image to linear RGB. Used in colorspace conversions.

    Args:
        image (Tensor[B, 3, H, W]):
            sRGB Image to be converted to linear RGB.

    Returns:
        linear_rgb (Tensor[B, 3, H, W]):
            linear RGB version of the image.
    """
    lin_rgb = torch.where(
        image > 0.04045,
        torch.pow(((image + 0.055) / 1.055), 2.4),
        image / 12.92
    )
    return lin_rgb


@dispatch(Tensor, (float, Tensor))
def rgb_to_rgba(image: Tensor, alpha_val: Union[float, Tensor]) -> Tensor:
    """Convert an image from RGB to RGBA.

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to RGBA.
        alpha_val (float, Tensor[B, 1, H, W]):
            A float number or tensor for the alpha value.

    Returns:
        rgba (Tensor[B, 4, H, W]):
            RGBA version of the image

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(f"`alpha_val` must be `float` or `Tensor`. "
                        f"But got: {type(alpha_val)}.")
  
    # Add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)
    a       = cast(Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))
    rgba = torch.cat([r, g, b, a], dim=-3)

    return rgba


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, (float, np.ndarray))
def rgb_to_rgba(image: np.ndarray, alpha_val: Union[float, np.ndarray]) -> np.ndarray:
    """Convert an image from RGB to RGBA.

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to RGBA.
        alpha_val (float, np.ndarray[B, 1, H, W]):
            A float number or tensor for the alpha value.

    Returns:
        rgba (np.ndarray[B, 4, H, W]):
            RGBA version of the image

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)


@dispatch(Tensor)
def rgba_to_bgr(image: Tensor) -> Tensor:
    """Convert an image from RGBA to BGR.

    Args:
        image (Tensor[B, 4, H, W]):
            RGBA Image to be converted to BGR.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"`image` must be a `Tensor`. But got: {type(image)}")
    if image.ndim < 3 or image.shape[-3] != 4:
        raise ValueError(f"`image` must have a shape of [*, 4, H, W]. "
                         f"But got: {image.shape}")

    # Convert to RGB first, then to BGR
    rgb = rgba_to_rgb(image)
    bgr = rgb_to_bgr(rgb)
    
    return bgr


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGBA to BGR.

    Args:
        image (np.ndarray[B, 4, H, W]):
            RGBA Image to be converted to BGR.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


@dispatch(Tensor)
def rgba_to_rgb(image: Tensor) -> Tensor:
    """Convert an image from RGBA to RGB.

    Args:
        image (Tensor[B, 4, H, W]):
            RGBA Image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    # Unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # Compute new channels
    a_one = torch.tensor(1.0) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b
    rgb   = torch.cat([r_new, g_new, b_new], dim=-3)

    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgba_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGBA to RGB.

    Args:
        image (np.ndarray[B, 4, H, W]):
            RGBA Image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_rgb")
class BgrToRgb(nn.Module):
    """Convert image from BGR to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_rgb(image)


@TRANSFORMS.register(name="bgr_to_rgba")
class BgrToRgba(nn.Module):
    """Convert an image from BGR to RGBA. Add an alpha channel to existing RGB
    image.

    Args:
        alpha_val (float, TensorOrArray[B, 1, H, W]):
            A float number or tensor for the alpha value.
 
    Notes:
        Current functionality is NOT supported by Torchscript.
    """

    # MARK: Magic Functions
    
    def __init__(self, alpha_val: Union[float, TensorOrArray]):
        super().__init__()
        self.alpha_val = alpha_val

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_rgba(image, self.alpha_val)


@TRANSFORMS.register(name="linear_rgb_to_rgb")
class LinearRgbToRgb(nn.Module):
    """Convert a linear RGB image to sRGB. Applies gamma correction to linear
    RGB values, at the end of colorspace conversions, to get sRGB.
   
    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb
        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
        [3] https://en.wikipedia.org/wiki/SRGB
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return linear_rgb_to_rgb(image)


@TRANSFORMS.register(name="rgb_to_bgr")
class RgbToBgr(nn.Module):
    """Convert an image from RGB to BGR. Image data is assumed to be in the
    range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_bgr(image)


@TRANSFORMS.register(name="rgb_to_linear_rgb")
class RgbToLinearRgb(nn.Module):
    """Convert an image from sRGB to linear RGB. Reverses the gamma correction
    of sRGB to get linear RGB values for colorspace conversions. Image data
    is assumed to be in the range of [0.0, 1.0].
 
    References:
        [1] https://stackoverflow.com/questions/35952564/convert-rgb-to-srgb
        [2] https://www.cambridgeincolour.com/tutorials/gamma-correction.htm
        [3] https://en.wikipedia.org/wiki/SRGB
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_linear_rgb(image)


@TRANSFORMS.register(name="rgb_to_rgba")
class RgbToRgba(nn.Module):
    """Convert an image from RGB to RGBA. Add an alpha channel to existing RGB
    image.

    Args:
        alpha_val (float, TensorOrArray[B, 1, H, W]):
            A float number or tensor for the alpha value.
 
    Notes:
        Current functionality is NOT supported by Torchscript.
    """

    # MARK: Magic Functions
    
    def __init__(self, alpha_val: Union[float, TensorOrArray]) -> None:
        super().__init__()
        self.alpha_val = alpha_val

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_rgba(image, self.alpha_val)


@TRANSFORMS.register(name="rgba_to_bgr")
class RgbaToBgr(nn.Module):
    """Convert an image from RGBA to BGR. Remove an alpha channel from BGR
    image.
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgba_to_bgr(image)


@TRANSFORMS.register(name="rgba_to_rgb")
class RgbaToRgb(nn.Module):
    """Convert an image from RGBA to RGB. Remove an alpha channel from RGB
    image.
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgba_to_rgb(image)
