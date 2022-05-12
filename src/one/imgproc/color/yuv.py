#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YUV color space.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from one.core import ListOrTuple2T
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.color.rgb import bgr_to_rgb
from one.imgproc.color.rgb import rgb_to_bgr
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "bgr_to_yuv",
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv422",
    "yuv420_to_rgb",
    "yuv422_to_rgb",
    "yuv_to_rgb",
    "BgrToYuv",
    "RgbToYuv",
    "RgbToYuv420",
    "RgbToYuv422",
    "Yuv420ToRgb",
    "Yuv422ToRgb",
    "YuvToBgr",
    "YuvToRgb",
]


# MARK: - Functional

@dispatch(Tensor)
def bgr_to_yuv(image: Tensor) -> Tensor:
    """Convert an BGR image to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            BGR Image to be converted to YUV.

    Returns:
        yuv (Tensor[B, 3, H, W]):
            YUV version of the image.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_yuv(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def bgr_to_yuv(image: np.ndarray) -> np.ndarray:
    """Convert an BGR image to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            BGR Image to be converted to YUV.

    Returns:
        yuv (np.ndarray[B, 3, H, W]):
            YUV version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


@dispatch(Tensor)
def rgb_to_yuv(image: Tensor) -> Tensor:
    """Convert an RGB image to YUV. Image data is assumed to be in the 
    range of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to YUV.

    Returns:
        yuv (Tensor[B, 3, H, W]):
            YUV version of the image.
    """
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]

    y   =  0.299 * r + 0.587 * g + 0.114 * b
    u   = -0.147 * r - 0.289 * g + 0.436 * b
    v   =  0.615 * r - 0.515 * g - 0.100 * b
    yuv = torch.stack([y, u, v], -3)
    
    return yuv


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def rgb_to_yuv(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB Image to be converted to YUV.

    Returns:
        yuv (np.ndarray[B, 3, H, W]):
            YUV version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def rgb_to_yuv420(image: Tensor) -> ListOrTuple2T[Tensor]:
    """Convert an RGB image to YUV 420 (subsampled). Image data is assumed
    to be in the range of [0.0, 1.0]. Input need to be padded to be evenly
    divisible by 2 horizontal and vertical. This function will output chroma
    siting [0.5, 0.5]

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to YUV.

    Returns:
        A Tensor containing the Y plane with shape [*, 1, H, W]
        A Tensor containing the UV planes with shape [*, 2, H/2, W/2]
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"`image` must be a `Tensor`. But got: {type(image)}.")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"`image` must have a shape of [*, 3, H, W]. "
                         f"But got: {image.shape}.")
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(f"`image` H, W must be evenly divisible by 2. "
                         f"But got: {image.shape}.")

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        F.avg_pool2d(yuvimage[..., 1:3, :, :], (2, 2))
    )


def rgb_to_yuv422(image: Tensor) -> ListOrTuple2T[Tensor]:
    """Convert an RGB image to YUV 422 (subsampled). Image data is assumed
    to be in the range of [0.0, 1.0]. Input need to be padded to be evenly
    divisible by 2 vertical. This function will output chroma siting (0.5)

    Args:
        image (Tensor[B, 3, H, W]):
            RGB Image to be converted to YUV.

    Returns:
       A Tensor containing the Y plane with shape [*, 1, H, W].
       A Tensor containing the UV planes with shape [*, 2, H, W/2].
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"`image` must be a Tensor. But got: {type(image)}.")
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"`image` must have a shape of [*, 3, H, W]. "
                         f"But got: {image.shape}.")
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(f"`image` H, W must be evenly divisible by 2. "
                         f"But got: {image.shape}.")

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        F.avg_pool2d(yuvimage[..., 1:3, :, :], (1, 2))
    )


def yuv420_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """Convert an YUV420 image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be
    padded to be evenly divisible by 2 horizontal and vertical. This function
    assumed chroma siting is [0.5, 0.5]

    Args:
        image_y (Tensor[B, 1, H, W]):
            Y (luma) Image plane to be converted to RGB.
        image_uv (Tensor[B, 2, H/2, W/2]):
            UV (chroma) Image planes to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    if not isinstance(image_y, Tensor):
        raise TypeError(f"`image` must be a `Tensor`. But got: {type(image_y)}.")
    if not isinstance(image_uv, Tensor):
        raise TypeError(f"`image` must be a `Tensor`. But got: {type(image_uv)}.")
    if len(image_y.shape) < 3 or image_y.shape[-3] != 1:
        raise ValueError(f"`image_y` must have a shape of [*, 1, H, W]. "
                         f"But got: {image_y.shape}.")
    if len(image_uv.shape) < 3 or image_uv.shape[-3] != 2:
        raise ValueError(f"`image_uv` must have a shape of [*, 2, H/2, W/2]. "
                         f"But got: {image_uv.shape}.")
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(f"`image_y` H, W must be evenly divisible by 2. "
                         f"But got: {image_y.shape}.")
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-2] / image_uv.shape[-2] != 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(f"`image_uv` H, W must be half the size of the luma "
                         f"plane. But got: {image_y.shape} and {image_uv.shape}.")

    # First upsample
    yuv444image = torch.cat([
        image_y, image_uv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """Convert an YUV422 image to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be
    padded to be evenly divisible by 2 vertical. This function assumed chroma
    siting is (0.5)

    Args:
        image_y (Tensor[B, 1, H, W]):
            Y (luma) Image plane to be converted to RGB.
        image_uv (Tensor[B, 2, H/2, W/2]):
            UV (luma) Image planes to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    if not isinstance(image_y, Tensor):
        raise TypeError(f"`image_y` must be a `Tensor`. But got: {type(image_y)}.")
    if not isinstance(image_uv, Tensor):
        raise TypeError(f"`image_y` must be `Tensor`. But got: {type(image_uv)}.")
    if len(image_y.shape) < 3 or image_y.shape[-3] != 1:
        raise ValueError(f"`image_y` must have a shape of [*, 1, H, W]. "
                         f"But got: {image_y.shape}.")
    if len(image_uv.shape) < 3 or image_uv.shape[-3] != 2:
        raise ValueError(f"`image_uv` must have a shape of [*, 2, H, W/2]. "
                         f"But got: {image_uv.shape}.")
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(f"`image_y` H, W must be evenly divisible by 2. "
                         f"But got: {image_y.shape}.")
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(f"`image_uv` W must be half the size of the luma "
                         f"plane. But got: {image_y.shape} and {image_uv.shape}")

    # First upsample
    yuv444image = torch.cat([
        image_y, image_uv.repeat_interleave(2, dim=-1)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


@dispatch(Tensor)
def yuv_to_bgr(image: Tensor) -> Tensor:
    """Convert an YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (Tensor[B, 3, H, W]):
            YUV Image to be converted to BGR.

    Returns:
        bgr (Tensor[B, 3, H, W]):
            BGR version of the image.
    """
    rgb = yuv_to_rgb(image)
    return rgb_to_bgr(rgb)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def yuv_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an YUV image to BGR. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (np.ndarray[B, 3, H, W]):
            YUV Image to be converted to BGR.

    Returns:
        bgr (np.ndarray[B, 3, H, W]):
            BGR version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR)


@dispatch(Tensor)
def yuv_to_rgb(image: Tensor) -> Tensor:
    """Convert an YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (Tensor[B, 3, H, W]):
            YUV Image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    y   = image[..., 0, :, :]
    u   = image[..., 1, :, :]
    v   = image[..., 2, :, :]

    r   = y + 1.14 * v  # coefficient for g is 0
    g   = y + -0.396 * u - 0.581 * v
    b   = y + 2.029 * u  # coefficient for b is 0
    rgb = torch.stack([r, g, b], -3)

    return rgb


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def yuv_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (np.ndarray[B, 3, H, W]):
            YUV Image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)


# MARK: - Modules

@TRANSFORMS.register(name="bgr_to_yuv")
class BgrToYuv(nn.Module):
    """Convert an image from BGR to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Reference:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return bgr_to_yuv(image)
    
    
@TRANSFORMS.register(name="rgb_to_yuv")
class RgbToYuv(nn.Module):
    """Convert an image from RGB to YUV. Image data is assumed to be in the
    range of [0.0, 1.0].

    Reference:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_yuv(image)


@TRANSFORMS.register(name="rgb_to_yuv420")
class RgbToYuv420(nn.Module):
    """Convert an image from RGB to YUV420. Image data is assumed to be in
    the range of [0.0, 1.0]. Width and Height evenly divisible by 2.

    Reference:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    # MARK: Forward Pass
    
    def forward(self, yuv_input: Tensor) -> ListOrTuple2T[Tensor]:
        return rgb_to_yuv420(yuv_input)


@TRANSFORMS.register(name="rgb_to_yuv422")
class RgbToYuv422(nn.Module):
    """Convert an image from RGB to YUV422. Image data is assumed to be in
    the range of [0.0, 1.0]. Width evenly disvisible by 2.
 
    Reference:
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    # MARK: Forward Pass
    
    def forward(self, yuv_input: Tensor) -> ListOrTuple2T[Tensor]:
        return rgb_to_yuv422(yuv_input)


@TRANSFORMS.register(name="yuv420_to_rgb")
class Yuv420ToRgb(nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Width and Height
    evenly divisible by 2.
    """

    # MARK: Forward Pass
    
    def forward(self, input_y: Tensor, input_uv: Tensor) -> Tensor:  # skipcq: PYL-R0201
        return yuv420_to_rgb(input_y, input_uv)


@TRANSFORMS.register(name="yuv422_to_rgb")
class Yuv422ToRgb(nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Width evenly
    divisible by 2.
    """

    # MARK: Forward Pass
    
    def forward(self, input_y: Tensor, input_uv: Tensor) -> Tensor:
        return yuv422_to_rgb(input_y, input_uv)


@TRANSFORMS.register(name="yuv_to_bgr")
class YuvToBgr(nn.Module):
    """Convert an image from YUV to Bgr. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.
    """
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return yuv_to_bgr(image)


@TRANSFORMS.register(name="yuv_to_rgb")
class YuvToRgb(nn.Module):
    """Convert an image from YUV to RGB. Image data is assumed to be in the
    range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return yuv_to_rgb(image)
