#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import sys
from typing import Union

import torch
from torch import Tensor

from one.core import assert_dict_contain_key
from one.core import Floats
from one.core import InterpolationMode
from one.core import ListOrTuple2T
from one.core import Transform
from one.vision.acquisition import get_image_hw
from one.vision.acquisition import get_num_channels
from one.vision.transformation.intensity import autocontrast


# MARK: - Functional -----------------------------------------------------------

def apply_transform_op(
    input        : Tensor,
    target       : Union[Tensor, None] = None,
    op_name      : str                 = "",
    magnitude    : Floats           = 0.0,
    interpolation: InterpolationMode   = InterpolationMode.NEAREST,
    fill         : Floats           = 0.0,
) -> Tensor:
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
        
        fill (Floats):

    """
    if op_name == "auto_contrast":
        input = autocontrast(input)
    elif op_name == "brightness":
        input = adjust_brightness(input, 1.0 + magnitude)
    elif op_name == "color":
        input = adjust_saturation(input, 1.0 + magnitude)
    elif op_name == "contrast":
        input = adjust_contrast(input, 1.0 + magnitude)
    elif op_name == "equalize":
        input = equalize(input)
    elif op_name == "hflip":
        input = F.hflip(input)
    elif op_name == "hflip_image_box":
        if target is None:
            raise ValueError("`target` must not be `None`.")
        input, target[:, 2:6] = hflip_image_box(input, target[:, 2:6])
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
    elif op_name == "identity":
        pass
    elif op_name == "invert":
        input = F.invert(input)
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
    elif op_name == "posterize":
        input = F.posterize(input, int(magnitude))
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
    elif op_name == "rotate":
        input = F.rotate(input, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "sharpness":
        input = F.adjust_sharpness(input, 1.0 + magnitude)
    elif op_name == "solarize":
        input = F.solarize(input, magnitude)
    elif op_name == "vflip":
        input = F.vflip(input)
    elif op_name == "vflip_image_box":
        if target is None:
            raise ValueError("`target` must not be None.")
        input, target[:, 2:6] = vflip_image_box(input, target[:, 2:6])
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
    else:
        raise ValueError(f"`op_name` must be recognized. But got: {op_name}.")
    
    if target is not None:
        return input, target
    else:
        return input


# MARK: - Module ---------------------------------------------------------------

class AutoAugment(Transform):
    """AutoAugment data augmentation method based on `"AutoAugment: Learning
    Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`.
    If the image is Tensor, it should be of type torch.uint8, and it is
    expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary
    number of leading dimensions.

    Args:
        policy (str):
            Augmentation policy. One of: [`imagenet`, `cifar10`, `svhn`].
            Default: `imagenet`.
        interpolation (InterpolationMode, str, int):
            Desired interpolation enum defined by `InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
        fill (Floats):
            Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Default: `0.0`.
        p (float, None):
            Probability of the image being adjusted. Default: `None` means
            process as normal.
    """

    cfgs = {
        "imagenet": [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("posterize", 0.4, 8   , 10)  , ("rotate"       , 0.6, 9   , 10)),
            (("solarize" , 0.6, 5   , 10)  , ("auto_contrast", 0.6, None, None)),
            (("equalize" , 0.8, None, None), ("equalize"     , 0.6, None, None)),
            (("posterize", 0.6, 7   , 10)  , ("posterize"    , 0.6, 6   , 10)),
            (("equalize" , 0.4, None, None), ("solarize"     , 0.2, 4   , 10)),
            (("equalize" , 0.4, None, None), ("rotate"       , 0.8, 8   , 10)),
            (("solarize" , 0.6, 3   , 10)  , ("equalize"     , 0.6, None, None)),
            (("posterize", 0.8, 5   , 10)  , ("equalize"     , 1.0, None, None)),
            (("rotate"   , 0.2, 3   , 10)  , ("solarize"     , 0.6, 8   , 10)),
            (("equalize" , 0.6, None, None), ("posterize"    , 0.4, 6   , 10)),
            (("rotate"   , 0.8, 8   , 10)  , ("color"        , 0.4, 0   , 10)),
            (("rotate"   , 0.4, 9   , 10)  , ("equalize"     , 0.6, None, None)),
            (("equalize" , 0.0, None, None), ("equalize"     , 0.8, None, None)),
            (("invert"   , 0.6, None, None), ("equalize"     , 1.0, None, None)),
            (("color"    , 0.6, 4   , 10)  , ("contrast"     , 1.0, 8   , 10)),
            (("rotate"   , 0.8, 8   , 10)  , ("color"        , 1.0, 2   , 10)),
            (("color"    , 0.8, 8   , 10)  , ("solarize"     , 0.8, 7   , 10)),
            (("sharpness", 0.4, 7   , 10)  , ("invert"       , 0.6, None, None)),
            (("hshear"   , 0.6, 5   , 10)  , ("equalize"     , 1.0, None, None)),
            (("color"    , 0.4, 0   , 10)  , ("equalize"     , 0.6, None, None)),
            (("equalize" , 0.4, None, None), ("solarize"     , 0.2, 4   , 10)),
            (("solarize" , 0.6, 5   , 10)  , ("auto_contrast", 0.6, None, None)),
            (("invert"   , 0.6, None, None), ("equalize"     , 1.0, None, None)),
            (("color"    , 0.6, 4   , 10)  , ("contrast"     , 1.0, 8   , 10)),
            (("equalize" , 0.8, None, None), ("equalize"     , 0.6, None, None)),
        ],
        "cifar10" : [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("invert"       , 0.1, None, None), ("contrast"     , 0.2, 6   , 10)),
            (("rotate"       , 0.7, 2   , 10)  , ("htranslate"   , 0.3, 9   , 10)),
            (("sharpness"    , 0.8, 1   , 10)  , ("sharpness"    , 0.9, 3   , 10)),
            (("vshear"       , 0.5, 8   , 10)  , ("vtranslate"   , 0.7, 9   , 10)),
            (("auto_contrast", 0.5, None, None), ("equalize"     , 0.9, None, None)),
            (("vshear"       , 0.2, 7   , 10)  , ("posterize"    , 0.3, 7   , 10)),
            (("color"        , 0.4, 3   , 10)  , ("brightness"   , 0.6, 7   , 10)),
            (("sharpness"    , 0.3, 9   , 10)  , ("brightness"   , 0.7, 9   , 10)),
            (("equalize"     , 0.6, None, None), ("equalize"     , 0.5, None, None)),
            (("contrast"     , 0.6, 7   , 10)  , ("sharpness"    , 0.6, 5   , 10)),
            (("color"        , 0.7, 7   , 10)  , ("htranslate"   , 0.5, 8   , 10)),
            (("equalize"     , 0.3, None, None), ("auto_contrast", 0.4, None, None)),
            (("vtranslate"   , 0.4, 3   , 10)  , ("sharpness"    , 0.2, 6   , 10)),
            (("brightness"   , 0.9, 6   , 10)  , ("color"        , 0.2, 8   , 10)),
            (("solarize"     , 0.5, 2   , 10)  , ("invert"       , 0.0, None, None)),
            (("equalize"     , 0.2, None, None), ("auto_contrast", 0.6, None, None)),
            (("equalize"     , 0.2, None, None), ("equalize"     , 0.6, None, None)),
            (("color"        , 0.9, 9   , 10)  , ("equalize"     , 0.6, None, None)),
            (("auto_contrast", 0.8, None, None), ("solarize"     , 0.2, 8   , 10)),
            (("brightness"   , 0.1, 3   , 10)  , ("color"        , 0.7, 0   , 10)),
            (("solarize"     , 0.4, 5   , 10)  , ("auto_contrast", 0.9, None, None)),
            (("vtranslate"   , 0.9, 9   , 10)  , ("vtranslate"   , 0.7, 9   , 10)),
            (("auto_contrast", 0.9, None, None), ("solarize"     , 0.8, 3   , 10)),
            (("equalize"     , 0.8, None, None), ("invert"       , 0.1, None, None)),
            (("vtranslate"   , 0.7, 9   , 10)  , ("auto_contrast", 0.9, None, None)),
        ],
        "svhn"    : [
            # (op_name, p, magnitude, num_magnitude_bins=10)
            (("hshear"  , 0.9, 4   , 10)  , ("invert"       , 0.2, None, None)),
            (("vshear"  , 0.9, 8   , 10)  , ("invert"       , 0.7, None, None)),
            (("equalize", 0.6, None, None), ("solarize"     , 0.6, 6   , 10)),
            (("invert"  , 0.9, None, None), ("equalize"     , 0.6, None, None)),
            (("equalize", 0.6, None, None), ("rotate"       , 0.9, 3   , 10)),
            (("hshear"  , 0.9, 4   , 10)  , ("auto_contrast", 0.8, None, None)),
            (("vshear"  , 0.9, 8   , 10)  , ("invert"       , 0.4, None, None)),
            (("vshear"  , 0.9, 5   , 10)  , ("solarize"     , 0.2, 6   , 10)),
            (("invert"  , 0.9, None, None), ("auto_contrast", 0.8, None, None)),
            (("equalize", 0.6, None, None), ("rotate"       , 0.9, 3   , 10)),
            (("hshear"  , 0.9, 4   , 10)  , ("solarize"     , 0.3, 3   , 10)),
            (("vshear"  , 0.8, 8   , 10)  , ("invert"       , 0.7, None, None)),
            (("equalize", 0.9, None, None), ("vtranslate"   , 0.6, 6   , 10)),
            (("invert"  , 0.9, None, None), ("equalize"     , 0.6, None, None)),
            (("contrast", 0.3, 3   , 10)  , ("rotate"       , 0.8, 4   , 10)),
            (("invert"  , 0.8, None, None), ("vtranslate"   , 0.0, 2   , 10)),
            (("vshear"  , 0.7, 6   , 10)  , ("solarize"     , 0.4, 8   , 10)),
            (("invert"  , 0.6, None, None), ("rotate"       , 0.8, 4   , 10)),
            (("vshear"  , 0.3, 7   , 10)  , ("htranslate"   , 0.9, 3   , 10)),
            (("hshear"  , 0.1, 6   , 10)  , ("invert"       , 0.6, None, None)),
            (("solarize", 0.7, 2   , 10)  , ("vtranslate"   , 0.6, 7   , 10)),
            (("vshear"  , 0.8, 4   , 10)  , ("invert"       , 0.8, None, None)),
            (("hshear"  , 0.7, 9   , 10)  , ("vtranslate"   , 0.8, 3   , 10)),
            (("vshear"  , 0.8, 5   , 10)  , ("auto_contrast", 0.7, None, None)),
            (("hshear"  , 0.7, 2   , 10)  , ("invert"       , 0.1, None, None)),
        ],
    }
    
    def __init__(
        self,
        policy       : str                = "imagenet",
        interpolation: InterpolationMode  = InterpolationMode.NEAREST,
        fill         : Floats          = 0.0,
        p            : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        assert_dict_contain_key(self.cfgs, policy)
        self.interpolation = interpolation
        self.fill          = fill
        self.transforms    = self.cfgs[policy]

    def _augmentation_space(
        self,
        num_bins  : int,
        image_size: ListOrTuple2T[int]
    ) -> dict[str, tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "auto_contrast": (torch.tensor(0.0), False),
            "brightness"   : (torch.linspace(0.0, 0.9,   num_bins), True),
            "color"        : (torch.linspace(0.0, 0.9,   num_bins), True),
            "contrast"     : (torch.linspace(0.0, 0.9,   num_bins), True),
            "equalize"     : (torch.tensor(0.0), False),
            "hflip"        : (torch.tensor(0.0), False),
            "vflip"        : (torch.tensor(0.0), False),
            "identity"     : (torch.tensor(0.0), False),
            "invert"       : (torch.tensor(0.0), False),
            "posterize"    : (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "rotate"       : (torch.linspace(0.0, 30.0,  num_bins), True),
            "sharpness"    : (torch.linspace(0.0, 0.9,   num_bins), True),
            "hshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "vshear"       : (torch.linspace(0.0, 0.3,   num_bins), True),
            "solarize"     : (torch.linspace(255.0, 0.0, num_bins), False),
            "htranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "vtranslate"   : (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        }
    
    def get_fill(self, input: Tensor) -> Union[Floats, None]:
        fill = self.fill
        if isinstance(fill, (int, float)):
            fill = [float(fill)] * get_num_channels(input)
        elif fill is not None:
            fill = [float(f) for f in fill]
        return fill
        
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        # NOTE: Fill
        fill = self.get_fill(input)
        
        # NOTE: Transform
        transform_id = int(torch.randint(len(self.transforms), (1,)).item())
        num_ops      = len(self.transforms[transform_id])
        probs        = torch.rand((num_ops,))
        signs        = torch.randint(2, (num_ops,))
        for i, (op_name, p, magnitude, num_magnitude_bins) in enumerate(
            self.transforms[transform_id]
        ):
            if probs[i] <= p:
                op_meta = self._augmentation_space(
                    num_magnitude_bins if num_magnitude_bins is not None else 10,
                    get_image_hw(input)
                )
                magnitudes, signed = op_meta[op_name]
                magnitude = (float(magnitudes[magnitude].item())
                             if magnitude is not None else 0.0)
                if signed and signs[i] == 0:
                    magnitude *= -1.0
                input = apply_transform_op(
                    input         = input,
                    op_name       = op_name,
                    magnitude     = magnitude,
                    interpolation = self.interpolation,
                    fill          = fill
                )
        
        # NOTE: Convert to tensor
        if self.to_tensor:
            input = to_tensor(input, normalize=True)
   
        return input


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
