#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements transformation functions"""

from __future__ import annotations

__all__ = [
    "Affine",
    "Hflip",
    "HomographyWarper",
    "PyrDown",
    "PyrUp",
    "Rescale",
    "Resize",
    "Rot180",
    "Rotate",
    "Scale",
    "ScalePyramid",
    "Shear",
    "Translate",
    "Vflip",
    "affine",
    "affine3d",
    "build_laplacian_pyramid",
    "build_pyramid",
    "center_crop",
    "center_crop3d",
    "crop_and_resize",
    "crop_and_resize3d",
    "crop_by_boxes",
    "crop_by_boxes3d",
    "crop_by_indices",
    "crop_by_transform_mat",
    "crop_by_transform_mat3d",
    "crop_divisible",
    "elastic_transform2d",
    "get_affine_matrix2d",
    "get_affine_matrix3d",
    "get_perspective_transform",
    "get_perspective_transform3d",
    "get_projective_transform",
    "get_rotation_matrix2d",
    "get_shear_matrix2d",
    "get_shear_matrix3d",
    "get_translation_matrix2d",
    "hflip",
    "homography_warp",
    "homography_warp3d",
    "invert_affine_transform",
    "pair_downsample",
    "projection_from_Rt",
    "pyrdown",
    "pyrup",
    "remap",
    "rescale",
    "resize",
    "resize_divisible",
]

from typing import Sequence

import cv2
import numpy as np
import PIL
import torch
from kornia.geometry import transform
# noinspection PyUnresolvedReferences
from kornia.geometry.transform import *
from plum import dispatch

from mon import core
from mon.nn import functional as F

console = core.console


# region Crop

def crop_divisible(image: PIL.Image, divisor: int = 32):
    """Make dimensions divisible by :param:`divisor`."""
    new_size = (image.size[0] - image.size[0] % divisor, image.size[1] - image.size[1] % divisor)
    box      = [
        int((image.size[0] - new_size[0]) / 2),
        int((image.size[1] - new_size[1]) / 2),
        int((image.size[0] + new_size[0]) / 2),
        int((image.size[1] + new_size[1]) / 2),
    ]
    image_cropped = image.crop(box=box)
    return image_cropped

# endregion


# region Downsample

def pair_downsample(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The image pair downsampler, which outputs two downsampled images of half
    the spatial resolution by averaging diagonal pixels in non-overlapping
    patches, as shown in the below figure:
    
                                     ---------------------
        ---------------------        | A1+D1/2 | A2+D2/2 |
        | A1 | B1 | A2 | B2 |        | A3+D3/2 | A4+D4/2 |
        | C1 | D1 | C2 | D2 |        ---------------------
        ---------------------  ===>
        | A3 | B3 | A4 | B4 |        ---------------------
        | C3 | D3 | C4 | D4 |        | B1+C1/2 | B2+C2/2 |
        ---------------------        | B3+C3/2 | B4+C4/2 |
                                     ---------------------
    
    References:
        `<https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing>`__
    """
    c       = input.shape[1]
    filter1 = torch.Tensor([[[[0, 0.5], [0.5, 0]]]]).to(input.dtype).to(input.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.Tensor([[[[0.5, 0], [0, 0.5]]]]).to(input.dtype).to(input.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(input, filter1, stride=2, groups=c)
    output2 = F.conv2d(input, filter2, stride=2, groups=c)
    return output1, output2

# endregion


# region Resize

@dispatch
def resize(
    image        : torch.Tensor,
    size         : int | Sequence[int],
    interpolation: str = "bilinear",
    **kwargs,
) -> torch.Tensor:
    """Resize an image using :mod:`kornia`.
    
    Args:
        image: An image tensor.
        size: The target size.
        interpolation: Algorithm used for upsampling: ``'nearest'`` | ``'linear'``
            | ``'bilinear'`` | ``'bicubic'`` | ``'trilinear'`` | ``'area'``.
    
    **kwargs (korina.geometry.transform.resize):
        - align_corners: interpolation flag.
        - side: Corresponding side if ``size`` is an integer. Can be one of ``'short'``, ``'long'``, ``'vert'``, or ``'horz'``.
        - antialias: if ``True``, then image will be filtered with Gaussian before downscaling. No effect for upscaling.
    
    **kwargs (cv2.resize):
        - fx: Scale factor along the horizontal axis.
        - fy: Scale factor along the vertical axis.
        - antialias: if ``True``, then image will be filtered with Gaussian before downscaling. No effect for upscaling.
    
    Returns:
        The resized image.
    """
    align_corners = kwargs.pop("align_corners", None)
    side          = kwargs.pop("side",          "short")
    antialias     = kwargs.pop("antialias",     False)
    return transform.resize(
        input         = image,
        size          = size,
        interpolation = interpolation,
        align_corners = align_corners,
        side          = side,
        antialias     = antialias,
    )


@dispatch
def resize(
    image        : np.ndarray,
    size         : int | Sequence[int],
    interpolation: int = cv2.INTER_LINEAR,
    **kwargs,
) -> np.ndarray:
    """Resize an image using :mod:`kornia`.
    
    Args:
        image: An image tensor.
        size: The target size.
        interpolation: Algorithm used for upsampling:
            - cv2.INTER_AREA: This is used when we need to shrink an image.
            - cv2.INTER_CUBIC: This is slow but more efficient.
            - cv2.INTER_LINEAR: This is primarily used when zooming is required. This is the default interpolation technique in OpenCV.

    **kwargs (cv2.resize):
        - fx: Scale factor along the horizontal axis.
        - fy: Scale factor along the vertical axis.
        - antialias: if ``True``, then image will be filtered with Gaussian before downscaling. No effect for upscaling.
    
    Returns:
        The resized image.
    """
    fx   = kwargs.pop("fx", None)
    fy   = kwargs.pop("fy", None)
    h, w = core.parse_hw(size)
    return cv2.resize(
        src           = image,
        dsize         = (w, h),
        fx            = fx,
        fy            = fy,
        interpolation = interpolation,
    )


@dispatch
def resize_divisible(image: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    """Resize an image to a size that is divisible by :param:`divisor`."""
    h, w  = core.get_image_size(image)
    h, w  = core.make_imgsz_divisible((h, w), divisor)
    image = resize(image, (h, w))
    return image


@dispatch
def resize_divisible(image: np.ndarray, divisor: int = 32) -> np.ndarray:
    """Resize an image to a size that is divisible by :param:`divisor`."""
    h, w  = core.get_image_size(image)
    h, w  = core.make_imgsz_divisible((h, w), divisor)
    image = resize(image, (w, h))
    return image

# endregion
