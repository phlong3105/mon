#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements resizing transformations. """

from __future__ import annotations

__all__ = [
    "letterbox_resize", "resize", "resize_crop",
]

import cv2
import numpy as np
import torch
from torchvision.transforms import (
    functional_tensor as functional_t,
)

from mon import foundation
from mon.coreimage import constant, util
from mon.coreimage.transform.affine import cropping
from mon.coreimage.typing import InterpolationModeType, Ints


def letterbox_resize(
    image     : np.ndarray,
    size      : Ints | None = 768,
    stride    : int         = 32,
    color     : Ints        = (114, 114, 114),
    auto      : bool        = True,
    scale_fill: bool        = False,
    scale_up  : bool        = True,
):
    """Resize qn image to a `stride`-pixel-multiple rectangle.
    
    Notes:
        For YOLOv5, stride = 32.
        For Scaled-YOLOv4, stride = 128
    
    References:
        https://github.com/ultralytics/yolov3/issues/232
    """
    old_size = util.get_image_size(image)
    
    if size is None:
        return image, None, None, None
    size = util.to_size(size)
    
    # Scale ratio (new / old)
    r = min(size[0] / old_size[0], size[1] / old_size[1])
    if not scale_up:  # only scale down, don't scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio     = r, r  # width, height ratios
    new_unpad = int(round(old_size[1] * r)), int(round(old_size[0] * r))
    dw, dh    = size[1] - new_unpad[0], size[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh    = 0.0, 0.0
        new_unpad = (size[1], size[0])
        ratio     = size[1] / old_size[1], size[0] / old_size[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if old_size[::-1] != new_unpad:  # resize
        image = cv2.resize(
            src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR
        )
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image       = cv2.copyMakeBorder(
        src        = image,
        top        = top,
        bottom     = bottom,
        left       = left,
        right      = right,
        borderType = cv2.BORDER_CONSTANT,
        value      = color
    )  # add border
    
    return image, ratio, (dw, dh)


def resize(
    image        : torch.Tensor,
    size         : Ints                  = None,
    interpolation: InterpolationModeType = "bilinear",
    antialias    : bool | None           = None,
) -> torch.Tensor:
    """Resize an image. Adapted from: :meth:`torchvision.transforms.functional.resize`
    
    Args:
        image: An image of shape [..., C, H, W] to be resized.
        size: An output size of shape [C, H, W].
        interpolation: An interpolation method. Defaults to “bilinear”.
        antialias: If True, perform antialias. Defaults to None.
        
    Returns:
        Resized image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    size  = util.to_size(size)  # H, W
    image = image.clone()
    if not isinstance(interpolation, constant.InterpolationMode):
        interpolation = constant.InterpolationMode.from_value(interpolation)
    if interpolation is constant.InterpolationMode.LINEAR:
        interpolation = constant.InterpolationMode.BILINEAR
    
    return functional_t.resize(
        img           = image,
        size          = foundation.to_list(size),  # H, W
        interpolation = str(interpolation.value),
        antialias     = antialias
    )


def resize_crop(
    image        : torch.Tensor,
    top          : int,
    left         : int,
    height       : int,
    width        : int,
    size         : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
) -> torch.Tensor:
    """Crop and resize an image. Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        image: An image of shape [..., C, H, W] to be resized.
        top: The vertical component of the top left corner of the crop box.
        left: The horizontal component of the top left corner of the crop box.
        height: The height of the crop box.
        width: The width of the crop box.
        size: An output size of shape [C, H, W]. Defaults to None.
        interpolation: An interpolation method. Defaults to “bilinear”.
        
    Returns:
        Resized crop image of shape [..., C, H, W].
    """
    image = cropping.crop(
        image   = image,
        top     = top,
        left    = left,
        height  = height,
        width   = width,
    )
    image = resize(
        image         = image,
        size          = size,
        interpolation = interpolation,
    )
    return image
