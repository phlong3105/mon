#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements rotating transformations. """

from __future__ import annotations

__all__ = [
    "rotate", "rotate_horizontal_flip", "rotate_image_box",
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
    """Rotates an image.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        angle: Angle to rotate the image.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to make it large enough to
            hold the entire rotated image. If False or omitted, make the output
            image the same size as the input image. Defaults to True. Note that
            the :param:`keep_shape` flag assumes rotation around the center and
            no translation.
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
        Rotated image of shape [..., C, H, W].
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
    """Rotates an image, then flips it vertically.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        angle: Angle to rotate the image.
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
        Rotated and flipped image of shape [..., C, H, W].
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
    image = flipping.horizontal_flip(image=image)
    return image


def rotate_image_box(
    image        : torch.Tensor,
    box          : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotates an image and a bounding box.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        box: Bounding box of shape [N, 4] to be rotated.
        angle: Angle to rotate the image.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
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
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Rotated image of shape [..., C, H, W].
        Rotated bounding box of shape [N, 4].
    """
    image, box = base.affine_image_box(
        image         = image,
        box           = box,
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
    return image, box


def rotate_vertical_flip(
    image        : torch.Tensor,
    angle        : float,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Rotates an image, then flips it vertically.
    
    Args:
        image: Image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        angle: Angle to rotate the image.
        center: Center of affine transformation. If None, use the center of the
            image. Defaults to None.
        interpolation: Desired interpolation mode. Default to "bilinear".
        keep_shape: If True, expands the output image to  make it large enough
            to hold the entire rotated image. If False or omitted, make the
            output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Horizontal translation (post-rotation translation).
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
        Rotated and flipped image of shape [..., C, H, W].
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
    image = flipping.vertical_flip(image=image)
    return image
