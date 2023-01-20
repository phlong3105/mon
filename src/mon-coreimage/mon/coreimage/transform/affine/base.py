#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base affine transformations."""

from __future__ import annotations

__all__ = [
    "affine", "affine_image_box", "pad",
]

import torch
from torch import nn
from torchvision.transforms import (
    functional as functional,
    functional_tensor as functional_t,
)

from mon import foundation
from mon.coreimage import constant, geometry, util
from mon.coreimage.typing import (
    Floats, InterpolationModeType, Ints, PaddingModeType,
)


# region Affine

def affine(
    image        : torch.Tensor,
    angle        : float,
    translate    : Ints,
    scale        : float,
    shear        : Floats,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
) -> torch.Tensor:
    """Apply an affine transformation on an image keeping image center
    invariant.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        angle: A rotation angle in degrees between -180 and 180, clockwise
            direction.
        translate: Horizontal and vertical translations (post-rotation
            translation).
        scale: An overall scale.
        shear: A shear angle value in degrees between -180 to 180, a clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center: The center of the affine transformation. If None, use the center
            of the image. Defaults to None.
        interpolation: An interpolation mode. Defaults to “bilinear”.
        keep_shape: If True, expands the output image to make it large enough
            to hold the entire rotated image. If False or omitted, make the
            output image the same size as the input image. Note that the
            :param:`keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill
              left/right/top/bottom bands respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: A padding mode. Defaults to “constant”.
        
    Returns:
        A transformed image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)

    assert isinstance(angle, int | float)
    if isinstance(angle, int):
        angle = float(angle)
    
    translate = foundation.to_list(translate)
    assert isinstance(translate, list) and len(translate) == 2
   
    if isinstance(scale, int):
        scale = float(scale)
    assert scale >= 0
    
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    assert isinstance(shear, list) and len(shear) == 2
    
    if not isinstance(interpolation, constant.InterpolationMode):
        interpolation = constant.InterpolationMode.from_value(interpolation)
    
    image  = image.clone()
    h, w   = util.get_image_size(image)
    center = (h * 0.5, w * 0.5) if center is None else center  # H, W
    center = list(center[::-1])  # W, H
    
    # If keep shape, find the new width and height bounds
    if not keep_shape:
        matrix  = functional._get_inverse_affine_matrix(
            center    = [0, 0],
            angle     = angle,
            translate = [0, 0],
            scale     = 1.0,
            shear     = [0.0, 0.0]
        )
        abs_cos = abs(matrix[0])
        abs_sin = abs(matrix[1])
        new_h   = int(h * abs_cos + w * abs_sin)
        new_w   = int(h * abs_sin + w * abs_cos)
        pad_h   = (new_h - h) / 2
        pad_w   = (new_w - w) / 2
        image   = pad(
            image        = image,
            padding      = (pad_h, pad_w),
            fill         = fill,
            padding_mode = padding_mode
        )
    
    translate_f = [1.0 * t for t in translate]
    matrix      = functional._get_inverse_affine_matrix(
        center    = [0, 0],
        angle     = angle,
        translate = translate_f,
        scale     = scale,
        shear     = shear
    )
    image = functional_t.affine(
        img           = image,
        matrix        = matrix,
        interpolation = interpolation.value,
        fill          = foundation.to_list(fill)
    )
    return image


def affine_image_box(
    image        : torch.Tensor,
    box          : torch.Tensor,
    angle        : float,
    translate    : Ints,
    scale        : float,
    shear        : Floats,
    center       : Ints | None           = None,
    interpolation: InterpolationModeType = "bilinear",
    keep_shape   : bool                  = True,
    fill         : Floats                = 0.0,
    padding_mode : PaddingModeType       = "constant",
    drop_ratio   : float                 = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an affine transformation on an image keeping image center 
    invariant.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        angle: A rotation angle in degrees between -180 and 180, clockwise
            direction.
        translate: Horizontal and vertical translations (post-rotation
            translation).
        scale: An overall scale.
        shear: A shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
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
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed image of shape [..., C, H, W].
        A transformed box of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    image_size = util.get_image_size(image)
    image = affine(
        image         = image,
        angle         = angle,
        translate     = translate,
        scale         = scale,
        shear         = shear,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        fill          = fill,
        padding_mode  = padding_mode,
    )
    box = geometry.affine_box(
        box        = box,
        image_size = image_size,
        angle      = angle,
        translate  = translate,
        scale      = scale,
        shear      = shear,
        center     = center,
        drop_ratio = drop_ratio,
    )
    return image, box

# endregion


# region Pad

def pad(
    image       : torch.Tensor,
    padding     : Floats,
    fill        : Floats          = 0.0,
    padding_mode: PaddingModeType = "constant",
) -> torch.Tensor:
    """Pads an image with fill value.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed, where ... means
            it can have an arbitrary number of leading dimensions.
        padding: Padding on each border. If a single int is provided this is
            used to pad all borders. If sequence of length 2 is provided this is
            the padding on left/right and top/bottom respectively. If a sequence
            of length 4 is provided this is the padding for the left, top, right
            and bottom borders respectively.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill
              left/right/top/bottom band respectively.
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode: One of: ["constant", "edge", "reflect", "symmetric"]
            - constant: pads with a constant value, this value is specified
              with fill.
            - edge: pads with the last value at the edge of the image. If input
              a 5D :class:`torch.Tensor`, the last 3 dimensions will be padded
              instead of the last 2.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3].
            Default: "constant".
        
    Returns:
        Padded image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)

    if isinstance(fill, (tuple, list)):
        fill = fill[0]
    assert isinstance(fill, int | float)

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if isinstance(padding, tuple):
        padding = list(padding)
    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError(
            f"`padding` must be an int or a 1, 2, or 4 element tuple. "
            f"But got: {len(padding)}."
        )
    
    padding = functional_t._parse_pad_padding(foundation.to_list(padding))

    if not isinstance(padding_mode, constant.PaddingMode):
        padding_mode = constant.PaddingMode.from_value(value=padding_mode)
    padding_mode = padding_mode.value
    
    if padding_mode == "edge":
        padding_mode = "replicate"  # Remap padding_mode str
    elif padding_mode == "symmetric":
        # route to another implementation
        return functional_t._pad_symmetric(image, padding)
    
    need_squeeze = False
    if image.ndim < 4:
        image        = image.unsqueeze(dim=0)
        need_squeeze = True
    out_dtype = image.dtype
    need_cast = False
    image     = image.clone()
    if (padding_mode != "constant") and image.dtype not in (torch.float32, torch.float64):
        # Here we temporarily cast the input tensor to float until the Pytorch
        # issue is resolved: https://github.com/pytorch/pytorch/issues/40763
        need_cast = True
        image     = image.to(torch.float32)
    if padding_mode in ("reflect", "replicate"):
        image = nn.functional.pad(image, padding, mode=padding_mode)
    else:
        image = nn.functional.pad(image, padding, mode=padding_mode, value=fill)
    if need_squeeze:
        image = image.squeeze(dim=0)
    if need_cast:
        image = image.to(out_dtype)
    return image

# endregion
