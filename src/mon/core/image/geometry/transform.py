#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformations.

This module implements transformation functions.
"""

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
import torch
from kornia.geometry import transform
from kornia.geometry.transform import *
from plum import dispatch
from torch.nn import functional as F

from mon.core.image import utils


# region Resize

def pair_downsample(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing
    """
    c       = image.shape[1]
    filter1 = torch.Tensor([[[[0, 0.5], [0.5, 0]]]]).to(image.dtype).to(image.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.Tensor([[[[0.5, 0], [0, 0.5]]]]).to(image.dtype).to(image.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


@dispatch
def resize(
    image        : torch.Tensor,
    size         : int | Sequence[int],
    interpolation: str = "bilinear",
    **kwargs,
) -> torch.Tensor:
    """Resize an image using :obj:`kornia`.
    
    Args:
        image: An image tensor.
        size: The target size.
        interpolation: Algorithm used for upsampling. One of:
            - ``'nearest'``
            - ``'linear'``
            - ``'bilinear'``
            - ``'bicubic'``
            - ``'trilinear'``
            -  ``'area'``
            Defaults: ``'bilinear'``.
    
    **kwargs (korina.geometry.transform.resize):
        - align_corners: interpolation flag.
        - side: Corresponding side if ``size`` is an integer. One of:
            + ``'short'``
            + ``'long'``
            + ``'vert'``
            + ``'horz'``
        - antialias: if ``True``, then image will be filtered with Gaussian
            before downscaling. No effect for upscaling.
    
    **kwargs (cv2.resize):
        - fx: Scale factor along the horizontal axis.
        - fy: Scale factor along the vertical axis.
        - antialias: If ``True``, then image will be filtered with Gaussian
            before downscaling. No effect for upscaling.
    
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
    """Resize an :obj:`image` using :obj:`kornia`.
    
    Args:
        image: An image tensor.
        size: The target size.
        interpolation: Algorithm used for upsampling:
            - cv2.INTER_AREA: This is used when we need to shrink an image.
            - cv2.INTER_CUBIC: This is slow but more efficient.
            - cv2.INTER_LINEAR: This is primarily used when zooming is required.
                This is the default interpolation technique in OpenCV.
    
    **kwargs (cv2.resize):
        - fx: Scale factor along the horizontal axis.
        - fy: Scale factor along the vertical axis.
        - antialias: If ``True``, then image will be filtered with Gaussian
            before downscaling. No effect for upscaling.
    
    Returns:
        The resized image.
    """
    fx   = kwargs.pop("fx", None)
    fy   = kwargs.pop("fy", None)
    h, w = utils.parse_hw(size)
    return cv2.resize(
        src           = image,
        dsize         = (w, h),
        fx            = fx,
        fy            = fy,
        interpolation = interpolation,
    )


def resize_divisible(
    image  : torch.Tensor | np.ndarray,
    divisor: int = 32
) -> torch.Tensor | np.ndarray:
    """Resize an :obj:`image` to a size that is divisible by :obj:`divisor`."""
    h, w  = utils.get_image_size(image)
    h, w  = utils.make_imgsz_divisible((h, w), divisor)
    image = resize(image, (w, h))
    return image

# endregion
