#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math
from typing import Optional

from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

from one.core import ListOrTupleAnyT
from one.core import TensorOrArray
from one.core import TensorsOrArrays
from one.imgproc import adjust_hsv
from one.imgproc import hflip_image_box
from one.imgproc import image_box_random_perspective
from one.imgproc import lowhighres_images_random_crop
from one.imgproc import vflip_image_box

__all__ = [
    "apply_transform_op",
]


# MARK: - Functional

def apply_transform_op(
    input        : TensorOrArray,
    target       : Optional[TensorOrArray] = None,
    op_name      : str                     = "",
    magnitude    : ListOrTupleAnyT[float]  = 0.0,
    interpolation: InterpolationMode       = InterpolationMode.NEAREST,
    fill         : Optional[list[float]]   = None,
) -> TensorsOrArrays:
    """Apply transform operation to the image and target.
    
    Args:
        input (Tensor[B, C, H, W]):
            Input to be transformed.
        target (Tensor[B, C, H, W], optional):
            Target to be transformed alongside with the input.
            If target is bounding boxes, labels[:, 2:6] in (x, y, x, y) format.
        op_name (str):
            Transform operation name.
        magnitude (ListOrTupleAnyT[float]):
        
        interpolation (InterpolationMode):
        
        fill (list[float], optional):

    """
    # NOTE: Adjust HSV
    if op_name == "adjust_hsv":
        if not isinstance(magnitude, (list, tuple)):
            raise TypeError(f"`magnitude` must be a list or tuple. "
                            f"But got: {type(magnitude)}.")
        if len(magnitude) != 3:
            raise ValueError(f"`magnitude` must have 3 elements. "
                             f"But got: {len(magnitude)}")
        input = adjust_hsv(
            input, h_factor=magnitude[0], s_factor=magnitude[1], v_factor=magnitude[2],
        )
    # NOTE: Auto Contrast
    elif op_name == "auto_contrast":
        input = F.autocontrast(input)
    # NOTE: Brightness
    elif op_name == "brightness":
        input = F.adjust_brightness(input, 1.0 + magnitude)
    # NOTE: Color
    elif op_name == "color":
        input = F.adjust_saturation(input, 1.0 + magnitude)
    # NOTE: Contrast
    elif op_name == "contrast":
        input = F.adjust_contrast(input, 1.0 + magnitude)
    # NOTE: Equalize
    elif op_name == "equalize":
        input = F.equalize(input)
    # NOTE: Hflip
    elif op_name == "hflip":
        input = F.hflip(input)
    # NOTE: Hflip Image Box
    elif op_name == "hflip_image_box":
        if target is None:
            raise ValueError("`target` must not be `None`.")
        input, target[:, 2:6] = hflip_image_box(input, target[:, 2:6])
    # NOTE: Hshear
    elif op_name == "hshear":
        input = F.affine(
            input,
            angle         = 0.0,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [math.degrees(magnitude), 0.0],
            interpolation = interpolation,
            fill          = fill,
        )
    # NOTE: Htranslate
    elif op_name == "htranslate":
        input = F.affine(
            input,
            angle         = 0.0,
            translate     = [int(magnitude), 0],
            scale         = 1.0,
            interpolation = interpolation,
            shear         = [0.0, 0.0],
            fill          = fill,
        )
    # NOTE: Identity
    elif op_name == "identity":
        pass
    # NOTE: Invert
    elif op_name == "invert":
        input = F.invert(input)
    # NOTE: Low-High Resolution Images Random Crop
    elif op_name == "lowhighres_images_random_crop":
        if not isinstance(magnitude, (list, tuple)):
            raise TypeError(f"`magnitude` must be a `list` or `tuple`. "
                            f"But got: {type(magnitude)}.")
        if len(magnitude) != 2:
            raise ValueError(f"`magnitude` must have 2 elements. "
                             f"But got: {len(magnitude)}")
        if target is None:
            raise ValueError("`target` must not be `None`.")
        input, target = lowhighres_images_random_crop(
            lowres  = input,
            highres = target,
            size    = magnitude[0],
            scale   = magnitude[1],
        )
    # NOTE: Posterize
    elif op_name == "posterize":
        input = F.posterize(input, int(magnitude))
    # NOTE: Random Box Perspective
    elif op_name == "image_box_random_perspective":
        if not isinstance(magnitude, (list, tuple)):
            raise TypeError(f"`magnitude` must be a `list` or `tuple`. "
                            f"But got: {type(magnitude)}.")
        if len(magnitude) != 5:
            raise ValueError(f"`magnitude` must have 5 elements. "
                             f"But got: {len(magnitude)}")
        if target is None:
            raise ValueError("`target` must not be None.")
        input, target = image_box_random_perspective(
            image       = input,
            box         = target,
            rotate      = magnitude[0],
            translate   = magnitude[1],
            scale       = magnitude[2],
            shear       = magnitude[3],
            perspective = magnitude[4],
        )
    # NOTE: Rotate
    elif op_name == "rotate":
        input = F.rotate(input, magnitude, interpolation=interpolation, fill=fill)
    # NOTE: Sharpness
    elif op_name == "sharpness":
        input = F.adjust_sharpness(input, 1.0 + magnitude)
    # NOTE: Solarize
    elif op_name == "solarize":
        input = F.solarize(input, magnitude)
    # NOTE: Vflip
    elif op_name == "vflip":
        input = F.vflip(input)
    # NOTE: Vflip Image Box
    elif op_name == "vflip_image_box":
        if target is None:
            raise ValueError("`target` must not be None.")
        input, target[:, 2:6] = vflip_image_box(input, target[:, 2:6])
    # NOTE: Vshear
    elif op_name == "vshear":
        input = F.affine(
            input,
            angle         = 0.0,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [0.0, math.degrees(magnitude)],
            interpolation = interpolation,
            fill          = fill,
        )
    # NOTE: Vtranslate
    elif op_name == "vtranslate":
        input = F.affine(
            input,
            angle         = 0.0,
            translate     = [0, int(magnitude)],
            scale         = 1.0,
            interpolation = interpolation,
            shear         = [0.0, 0.0],
            fill          = fill,
        )
    # NOTE: Error
    else:
        raise ValueError(f"`op_name` must be recognized. But got: {op_name}.")
    
    if target is not None:
        return input, target
    else:
        return input
