#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements translating transformations. """

from __future__ import annotations

__all__ = [
    "translate", "translate_horizontal", "translate_image_bbox",
    "translate_image_bbox_horizontal", "translate_image_bbox_vertical",
    "translate_vertical",
]

import torch

from mon.coreimage.transform.affine import base
from mon.coreimage.typing import (
    Floats, InterpolationModeType, Ints, PaddingModeType,
)


def translate(
    image        : torch.Tensor,
    magnitude    : Ints,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Translate an image.
    
    Args:
        image: An image of shape [..., C, H, W].
        magnitude: Horizontal and vertical translations (post-rotation
            translation).
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
        A transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = magnitude,
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image


def translate_horizontal(
    image        : torch.Tensor,
    magnitude    : int,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Translate an image in horizontal direction.
    
    Args:
        image: An image of shape [..., C, H, W].
        magnitude: A horizontal translation (post-rotation translation)
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
        A transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = [magnitude, 0],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image


def translate_vertical(
    image        : torch.Tensor,
    magnitude    : int,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Translate an image in vertical direction.
    
    Args:
        image: An image of shape [..., C, H, W].
        magnitude: A vertical translation (post-rotation translation)
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
        A transformed image of shape [..., C, H, W].
    """
    image = base.affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, magnitude],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    return image


def translate_image_bbox(
    image        : torch.Tensor,
    bbox         : torch.Tensor,
    magnitude    : Ints,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate an image and a bounding bbox.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling
        -translation/
        
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        magnitude: Horizontal and vertical translations (post-rotation
            translation).
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
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed image of shape [..., C, H, W].
        A translated bbox of shape [N, 4].
    """
    image, bbox = base.affine_image_bbox(
        image         = image,
        bbox          = bbox,
        angle         = 0.0,
        translate     = magnitude,
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


def translate_image_bbox_horizontal(
    image        : torch.Tensor,
    bbox         : torch.Tensor,
    magnitude    : int,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate an image and a bounding bbox in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling
        -translation/
        
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        magnitude: A horizontal translation (post-rotation translation).
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
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed image of shape [..., C, H, W].
        A translated bbox of shape [N, 4].
    """
    image, bbox = base.affine_image_bbox(
        image         = image,
        bbox          = bbox,
        angle         = 0.0,
        translate     = [magnitude, 0],
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


def translate_image_bbox_vertical(
    image        : torch.Tensor,
    bbox         : torch.Tensor,
    magnitude    : int,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate an image and a bounding bbox in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling
        -translation/
        
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        magnitude: Vertical translation (post-rotation translation).
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
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed image of shape [..., C, H, W].
        A translated bbox of shape [N, 4].
    """
    image, bbox = base.affine_image_bbox(
        image         = image,
        bbox          = bbox,
        angle         = 0.0,
        translate     = [0, magnitude],
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
