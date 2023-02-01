#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements rotating transformations. """

from __future__ import annotations

__all__ = [
    "rotate", "rotate_horizontal_flip", "rotate_image_bbox",
    "rotate_vertical_flip",
]

import torch

from mon.coreimage.transform.affine import base, flipping
from mon.coreimage.typing import (
    Floats, InterpolationModeType, Ints, PaddingModeType,
)


def rotate(
    image        : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Rotate an image.
    
    Args:
        image: An image of shape [..., C, H, W].
        angle: Angle to rotate the image.
        center: The center of the affine transformation. If None, use the center
            of the image. Defaults to None.
        interpolation: An interpolation mode. Defaults to “bilinear”.
        keep_shape: If True, expands the output image to make it large enough to
            hold the entire rotated image. If False or omitted, make the output
            image the same size as the input image. Defaults to True. Note that
            the :param:`keep_shape` flag assumes rotation around the center and
            no translation.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill 
              left/right/top/bottom band respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: A padding mode. Defaults to “constant”.
        
    Returns:
        A rotated image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    return base.affine(
        image         = image,
        angle         = angle,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )


def rotate_horizontal_flip(
    image        : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Rotate an image, then flip it vertically.
    
    Args:
        image: An image of shape [..., C, H, W].
        angle: Angle to rotate the image.
        center: The center of the affine transformation. If None, use the center
            of the image. Defaults to None.
        interpolation: An interpolation mode. Defaults to “bilinear”.
        keep_shape: If True, expands the output image to make it large enough to
            hold the entire rotated image. If False or omitted, make the output
            image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill
              left/right/top/bottom band respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: A padding mode. Defaults to “constant”.
        
    Returns:
        A rotated and flipped image of shape [..., C, H, W].
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    image = flipping.flip_horizontal(image=image)
    return image


def rotate_image_bbox(
    image        : torch.Tensor,
    bbox         : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate an image and a bounding bbox.
    
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        angle: Angle to rotate the image.
        center: The center of the affine transformation. If None, use the center
            of the image. Defaults to None.
        interpolation: An interpolation mode. Defaults to “bilinear”.
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill
              left/right/top/bottom band respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: A padding mode. Defaults to “constant”.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A rotated image of shape [..., C, H, W].
        A rotated bounding bbox of shape [N, 4].
    """
    image, bbox = base.affine_image_bbox(
        image         = image,
        bbox          = bbox,
        angle         = angle,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
    )
    return image, bbox


def rotate_vertical_flip(
    image        : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Rotate an image, then flip it vertically.
    
    Args:
        image: An image of shape [..., C, H, W].
        angle: Angle to rotate the image.
        center: The center of the affine transformation. If None, use the center
            of the image. Defaults to None.
        interpolation: An interpolation mode. Defaults to “bilinear”.
        keep_shape: If True, expands the output image to make it large enough
            to hold the entire rotated image. If False or omitted, make the
            output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Horizontal translation (post-rotation translation).
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill
              left/right/top/bottom band respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: A padding mode. Defaults to “constant”.
        
    Returns:
        A rotated and flipped image of shape [..., C, H, W].
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    image = flipping.flip_vertical(image=image)
    return image
