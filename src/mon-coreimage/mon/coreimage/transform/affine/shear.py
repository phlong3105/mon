#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements shearing transformations. """

from __future__ import annotations

__all__ = [
    "horizontal_shear", "shear", "shear_image_box", "vertical_shear",
]

import torch

from mon.coreimage.transform.affine import base
from mon.coreimage.typing import (
    Floats, InterpolationModeType, Ints, PaddingModeType,
)


def horizontal_shear(
    image        : torch.Tensor,
    magnitude    : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Shears an image horizontally.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        magnitude: Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to make it large enough to
            hold the entire rotated image. If False or omitted, make the output
            image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: Desired padding mode. Defaults to "constant".
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [magnitude, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image


def shear(
    image        : torch.Tensor,
    magnitude    : Floats,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Shears an image.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        magnitude: Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to make it large enough to
            hold the entire rotated image. If False or omitted, make the output
            image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: Desired padding mode. Defaults to "constant".
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image


def shear_image_box(
    image        : torch.Tensor,
    box          : torch.Tensor,
    magnitude    : Floats,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shears an image and a bounding box.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
        it can have an arbitrary number of leading dimensions.
        box: Bounding box of shape [N, 4] to be sheared.
        magnitude: Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to  make it large enough
            to hold the entire rotated image. If False or omitted, make the
            output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: Desired padding mode. Defaults to "constant".
        drop_ratio: If the fraction of a bounding box left in the image
            after being clipped is less than :param:`drop_ratio` the bounding
            box is dropped. If :param:`drop_ratio` == 0, don't drop any bounding
            boxes. Defaults to 0.0.
        
    Returns:
        Rotated image of shape [..., C, H, W].
        Rotated bounding box of shape [N, 4].
    """
    image, box = base.affine_image_box(
        image         = image,
        box           = box,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
        drop_ratio    = drop_ratio,
    )
    return image, box


def vertical_shear(
    image        : torch.Tensor,
    magnitude    : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Shears an image vertically.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        magnitude: Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to  make it large enough
            to hold the entire rotated image. If False or omitted, make the
            output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: Desired padding mode. Defaults to "constant".
        
    Returns:
        Transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0.0, magnitude],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image
