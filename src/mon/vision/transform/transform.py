#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image transformations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import PIL.Image
import torch

from mon import coreimage as ci, coreml
from mon.vision import constant
from mon.vision.typing import (
    Float3T, Floats, Image, InterpolationModeType, Ints, PaddingModeType,
)


# region Affine

@constant.TRANSFORM.register(name="affine")
class Affine(coreml.Transform):
    
    def __init__(
        self,
        angle        : float,
        translate    : Ints,
        scale        : float,
        shear        : Floats,
        center       : Ints  | None          = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.translate     = translate
        self.scale         = scale
        self.shear         = shear
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.affine(
            image         = input,
            angle         = self.angle,
            translate     = self.translate,
            scale         = self.scale,
            shear         = self.shear,
            center        = self.center,
            interpolation = self.interpolation,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.affine(
            image         = target,
            angle         = self.angle,
            translate     = self.translate,
            scale         = self.scale,
            shear         = self.shear,
            center        = self.center,
            interpolation = self.interpolation,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target

# endregion


# region Color

@constant.TRANSFORM.register(name="rgb_to_bgr")
class RGBToBGR(coreml.Transform):
   
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_bgr(image=input)
        target = ci.rgb_to_bgr(image=target) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgb_to_grayscale")
class RGBToGrayscale(coreml.Transform):
    
    def __init__(
        self,
        rgb_weights: Float3T | torch.Tensor | None,
        p          : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.rgb_weights = rgb_weights
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_grayscale(image=input,  rgb_weights=self.rgb_weights)
        target = ci.rgb_to_grayscale(image=target, rgb_weights=self.rgb_weights)\
            if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgb_to_hsv")
class RGBToHSV(coreml.Transform):
    
    def __init__(
        self,
        eps: float = 1e-8,
        p  : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.eps = eps
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_hsv(image=input,  eps=self.eps)
        target = ci.rgb_to_hsv(image=target, eps=self.eps) \
            if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="rgb_to_lab")
class RGBToLab(coreml.Transform):
   
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_lab(image=input)
        target = ci.rgb_to_lab(image=target) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="rgb_to_linear_rgb")
class RGBToLinearRGB(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_linear_rgb(image=input)
        target = ci.rgb_to_linear_rgb(image=target) \
            if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgb_to_luv")
class RGBToLUV(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_luv(image=input)
        target = ci.rgb_to_luv(image=target) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="rgb_to_rgba")
class RGBToRGBA(coreml.Transform):
    
    def __init__(
        self,
        alpha_val: float | torch.Tensor,
        p        : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.alpha_val = alpha_val
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_rgba(image=input,  alpha_val=self.alpha_val)
        target = ci.rgb_to_rgba(image=target, alpha_val=self.alpha_val) \
            if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="rgb_to_xyz")
class RGBToXYZ(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_xyz(image=input)
        target = ci.rgb_to_xyz(image=target) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgb_to_ycrcb")
class RGBToYCrCb(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_ycrcb(image=input)
        target = ci.rgb_to_ycrcb(image=target) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="rgb_to_yuv")
class RGBToYUV(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_yuv(image=input)
        target = ci.rgb_to_yuv(image=target) if target is not None else None
        return input, target
        
        
@constant.TRANSFORM.register(name="rgb_to_yuv420")
class RGBToYUV420(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_yuv420(image=input)
        target = ci.rgb_to_yuv420(image=target) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="rgb_to_yuv422")
class RGBToYUV422(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgb_to_yuv422(image=input)
        target = ci.rgb_to_yuv422(image=target) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgba_to_bgr")
class RGBAToBGR(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgba_to_bgr(image=input)
        target = ci.rgba_to_bgr(image=target) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rgba_to_rgb")
class RGBaToRGB(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.rgba_to_rgb(image=input)
        target = ci.rgba_to_rgb(image=target) if target is not None else None
        return input, target
        
# endregion


# region Conversion

@constant.TRANSFORM.register(name="denormalize")
class Denormalize(coreml.Transform):

    def __init__(
        self,
        min    : float        = 0.0,
        max    : float        = 1.0,
        new_min: float        = 0.0,
        new_max: float        = 255.0,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.min     = min
        self.max     = max
        self.new_min = new_min
        self.new_max = new_max
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.normalize_by_range(
            image   = input,
            min     = self.min,
            max     = self.max,
            new_min = self.new_min,
            new_max = self.new_max,
        )
        target  = ci.normalize_by_range(
            image   = target,
            min     = self.min,
            max     = self.max,
            new_min = self.new_min,
            new_max = self.new_max,
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="normalize")
class Normalize(coreml.Transform):

    def __init__(
        self,
        min    : float        = 0.0,
        max    : float        = 255.0,
        new_min: float        = 0.0,
        new_max: float        = 1.0,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.min     = min
        self.max     = max
        self.new_min = new_min
        self.new_max = new_max
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.normalize_by_range(
            image   = input,
            min     = self.min,
            max     = self.max,
            new_min = self.new_min,
            new_max = self.new_max,
        )
        target  = ci.normalize_by_range(
            image   = target,
            min     = self.min,
            max     = self.max,
            new_min = self.new_min,
            new_max = self.new_max,
        ) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="to_image")
class ToImage(coreml.Transform):

    def __init__(
        self,
        keepdim    : bool = True,
        denormalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keepdim     = keepdim
        self.denormalize = denormalize
        
    def forward(
        self,
        input : torch.torch.Tensor,
        target: torch.torch.Tensor | None = None,
    ) -> tuple[np.ndrray, np.ndrray | None]:
        input = ci.to_image(
            image       = input,
            keepdim     = self.keepdim,
            denormalize = self.denormalize
        )
        target = ci.to_image(
            image       = target,
            keepdim     = self.keepdim,
            denormalize = self.denormalize
        ) if target is not None else None


@constant.TRANSFORM.register(name="to_tensor")
class ToTensor(coreml.Transform):

    def __init__(
        self,
        keepdim  : bool = False,
        normalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keepdim   = keepdim
        self.normalize = normalize
        
    def forward(
        self,
        input : Image | PIL.Image,
        target: Image | PIL.Image | None = None,
    ) -> tuple[torch.torch.Tensor, torch.torch.Tensor | None]:
        input = ci.to_tensor(
            image     = input,
            keepdim   = self.keepdim,
            normalize = self.normalize
        )
        target = ci.to_tensor(
            image     = input,
            keepdim   = self.keepdim,
            normalize = self.normalize
        ) if target is not None else None
        return input, target

# endregion


# region Crop

@constant.TRANSFORM.register(name="center_crop")
class CenterCrop(coreml.Transform):

    def __init__(
        self,
        output_size: Ints,
        p          : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.output_size = output_size
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.center_crop(image=input,  output_size=self.output_size)
        target = ci.center_crop(image=target, output_size=self.output_size) \
            if target is not None else None
        return input, target
    
    
@constant.TRANSFORM.register(name="crop")
class Crop(coreml.Transform):

    def __init__(
        self,
        top    : int,
        left   : int,
        height : int,
        width  : int,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.top     = top
        self.left    = left
        self.height  = height
        self.width   = width
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.crop(
            image  = input,
            top    = self.top,
            left   = self.left,
            height = self.height,
            width  = self.width,
        )
        target = ci.crop(
            image  = target,
            top    = self.top,
            left   = self.left,
            height = self.height,
            width  = self.width,
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="five_crop")
class FiveCrop(coreml.Transform):

    def __init__(
        self,
        size : Ints,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size = size
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    ]:
        input  = ci.five_crop(image=input,  size=self.size)
        target = ci.five_crop(image=target, size=self.size) \
            if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="ten_crop")
class TenCrop(coreml.Transform):

    def __init__(
        self,
        size : Ints,
        vflip: bool         = False,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size  = size
        self.vflip = vflip
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None
    ]:
        input  = ci.ten_crop(image=input,  size=self.size, vflip=self.vflip)
        target = ci.ten_crop(image=target, size=self.size, vflip=self.vflip) \
            if target is not None else None
        return input, target
        
# endregion


# region Flip

@constant.TRANSFORM.register(name="hflip")
@constant.TRANSFORM.register(name="horizontal_flip")
class HorizontalFlip(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.flip_horizontal(image=input)
        target = ci.flip_horizontal(image=target) \
            if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="hflip_image_box")
@constant.TRANSFORM.register(name="horizontal_flip_image_box")
class HorizontalFlipImageBox(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.flip_image_bbox_horizontal(image=input, bbox=target)


@constant.TRANSFORM.register(name="vflip")
@constant.TRANSFORM.register(name="vertical_flip")
class VerticalFlip(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
      
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.flip_vertical(image=input)
        target = ci.flip_vertical(image=target) if target is not None else None
        return input, target
    
    
@constant.TRANSFORM.register(name="vflip_image_box")
@constant.TRANSFORM.register(name="vertical_flip_image_box")
class VerticalFlipImageBox(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.flip_image_bbox_vertical(image=input, bbox=target)

# endregion


# region Intensity

@constant.TRANSFORM.register(name="adjust_brightness")
class AdjustBrightness(coreml.Transform):
   
    def __init__(
        self,
        brightness_factor: float,
        p                : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness_factor = brightness_factor
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci. adjust_brightness(
            image             = input,
            brightness_factor = self.brightness_factor
        )
        target = ci.adjust_brightness(
            image             = target,
            brightness_factor = self.brightness_factor
        ) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="adjust_contrast")
class AdjustContrast(coreml.Transform):

    def __init__(
        self,
        contrast_factor: float,
        p              : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.contrast_factor = contrast_factor
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.adjust_contrast(
            image           = input,
            contrast_factor = self.contrast_factor
        )
        target = ci.adjust_contrast(
            image           = target,
            contrast_factor = self.contrast_factor
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="adjust_gamma")
class AdjustGamma(coreml.Transform):

    def __init__(
        self,
        gamma: float,
        gain : float        = 1.0,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.gamma = gamma
        self.gain  = gain
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.adjust_gamma(
            image = input,
            gamma = self.gamma,
            gain  = self.gain
        )
        target = ci.adjust_gamma(
            image = target,
            gamma = self.gamma,
            gain  = self.gain
        ) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="adjust_hue")
class AdjustHue(coreml.Transform):

    def __init__(
        self,
        hue_factor: float,
        p         : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.hue_factor = hue_factor
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.adjust_hue(
            image      = input,
            hue_factor = self.hue_factor
        )
        target = ci.adjust_hue(
            image      = target,
            hue_factor = self.hue_factor
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="adjust_saturation")
class AdjustSaturation(coreml.Transform):
    
    def __init__(
        self,
        saturation_factor: float,
        p                : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.saturation_factor = saturation_factor
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.adjust_saturation(
            image             = input,
            saturation_factor = self.saturation_factor
        )
        target = ci.adjust_saturation(
            image             = target,
            saturation_factor = self.saturation_factor
        ) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="adjust_sharpness")
class AdjustSharpness(coreml.Transform):
    
    def __init__(
        self,
        sharpness_factor: float,
        p               : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.sharpness_factor = sharpness_factor
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.adjust_sharpness(
            image            = input,
            sharpness_factor = self.sharpness_factor
        )
        target = ci.adjust_sharpness(
            image            = target,
            sharpness_factor = self.sharpness_factor
        ) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="autocontrast")
class AutoContrast(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.autocontrast(image=input)
        target = ci.autocontrast(image=target) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="color_jitter")
class ColorJitter(coreml.Transform):

    def __init__(
        self,
        brightness: Floats | None = 0.0,
        contrast  : Floats | None = 0.0,
        saturation: Floats | None = 0.0,
        hue       : Floats | None = 0.0,
        p         : float  | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness: list[float] = self._check_input(brightness, "brightness")
        self.contrast  : list[float] = self._check_input(contrast,   "contrast")
        self.saturation: list[float] = self._check_input(saturation, "saturation")
        self.hue       : list[float] = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
    
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue})"
        )
        return s
        
    @torch.jit.unused
    def _check_input(
        self,
        value             : Any,
        name              : str,
        center            : int   = 1,
        bound             : tuple = (0, float("inf")),
        clip_first_on_zero: bool  = True
    ) -> list[float] | None:
        if isinstance(value, int | float):
            assert value >= 0
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, tuple | list):
            assert len(value) == 2
            assert bound[0] <= value[0] <= bound[1]
            assert bound[0] <= value[1] <= bound[1]
        else:
            raise TypeError(
                f"{name} must be a single number or a sequence of 2 numbers. "
                f"But got: {value}."
            )

        # If value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    @staticmethod
    def get_params(
        brightness: Floats | None,
        contrast  : Floats | None,
        saturation: Floats | None,
        hue       : Floats | None,
    ) -> tuple[
        torch.Tensor,
        float | None,
        float | None,
        float | None,
        float | None
    ]:
        fn_idx = torch.randperm(4)
        b = None if brightness is None \
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast   is None \
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None \
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None \
            else float(torch.empty(1).uniform_(hue[0], hue[1]))
        return fn_idx, b, c, s, h
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fn_idx, \
            brightness_factor, \
            contrast_factor, \
            saturation_factor, \
            hue_factor \
            = self.get_params(
                brightness = self.brightness,
                contrast   = self.contrast,
                saturation = self.saturation,
                hue        = self.hue
            )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                input  = ci.adjust_brightness(input,  brightness_factor)
                target = ci.adjust_brightness(target, brightness_factor) \
                    if target is not None else None
            elif fn_id == 1 and contrast_factor is not None:
                input  = ci.adjust_contrast(input, contrast_factor)
                target = ci.adjust_contrast(target, contrast_factor) \
                    if target is not None else None
            elif fn_id == 2 and saturation_factor is not None:
                input  = ci.adjust_saturation(input,  saturation_factor)
                target = ci.adjust_saturation(target, saturation_factor) \
                    if target is not None else None
            elif fn_id == 3 and hue_factor is not None:
                input  = ci.adjust_hue(input,  hue_factor)
                target = ci.adjust_hue(target, hue_factor) \
                    if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="erase")
class Erase(coreml.Transform):

    def __init__(
        self,
        i: int,
        j: int,
        h: int,
        w: int,
        v: torch.Tensor,
        p: float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.i = i
        self.j = j
        self.h = h
        self.w = w
        self.v = v
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.erase(
            image = input,
            i     = self.i,
            j     = self.j,
            h     = self.h,
            w     = self.w,
            v     = self.v,
        )
        target = ci.erase(
            image = target,
            i     = self.i,
            j     = self.j,
            h     = self.h,
            w     = self.w,
            v     = self.v,
        ) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="equalize")
class Equalize(coreml.Transform):
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.equalize(image=input)
        target = ci.equalize(image=target) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="invert")
class Invert(coreml.Transform):

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
   
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ci.invert(image=input), \
               ci.invert(image=target) if target is not None else None


@constant.TRANSFORM.register(name="posterize")
class Posterize(coreml.Transform):
    
    def __init__(
        self,
        bits: int,
        p   : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.bits = bits
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return ci.posterize(image=input,  bits=self.bits), \
               ci.posterize(image=target, bits=self.bits) \
                   if target is not None else None


@constant.TRANSFORM.register(name="random_erase")
class RandomErase(coreml.Transform):

    def __init__(
        self,
        scale  : Floats                           = (0.02, 0.33),
        ratio  : Floats                           = (0.3, 3.3),
        value  : int | float | str | tuple | list = 0,
        p      : float | None                     = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        if not isinstance(value, (int, float, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence."
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'.")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence.")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence.")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max).")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1.")
        
        self.scale   = scale
        self.ratio   = ratio
        self.value   = value
        
    @staticmethod
    def get_params(
        image: torch.Tensor,
        scale: Floats,
        ratio: Floats,
        value: list[float] | None = None
    ) -> tuple[int, int, int, int, torch.Tensor]:
        img_c, img_h, img_w = ci.get_image_shape(image)
        area                = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area   = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, image
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value
    
        if value is not None and not (len(value) in (1, input.shape[-3])):
            raise ValueError(
                f"If value is a sequence, it should have either a single value "
                f"or {input.shape[-3]} (number of input channels)."
            )
    
        x, y, h, w, v = self.get_params(
            image = input,
            scale = self.scale,
            ratio = self.ratio,
            value = value
        )
        return \
            ci.erase(
                image   = input,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
            ), \
            ci.erase(
                image   = target,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
            ) if target is not None else None
    

@constant.TRANSFORM.register(name="solarize")
class Solarize(coreml.Transform):
    
    def __init__(
        self,
        threshold: float,
        p        : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.threshold = threshold
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.solarize(image=input,  threshold=self.threshold)
        target = ci.solarize(image=target, threshold=self.threshold)\
            if target is not None else None
        return input, target

# endregion


# region Resize

@constant.TRANSFORM.register(name="resize")
class Resize(coreml.Transform):

    def __init__(
        self,
        size         : Ints,
        interpolation: InterpolationModeType = "bilinear",
        antialias    : bool  | None          = None,
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size          = size
        self.interpolation = interpolation
        self.antialias     = antialias
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.resize(
                image         = input,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
        )
        target = ci.resize(
                image         = target,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
        ) if target is not None else None
        return input, target
        

@constant.TRANSFORM.register(name="resize_crop")
class ResizeCrop(coreml.Transform):

    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: InterpolationModeType = "bilinear",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.resize_crop(
            image         = input,
            top           = self.top,
            left          = self.left,
            height        = self.height,
            width         = self.width,
            size          = self.size,
            interpolation = self.interpolation,
        )
        target = ci.resize_crop(
            image         = target,
            top           = self.top,
            left          = self.left,
            height        = self.height,
            width         = self.width,
            size          = self.size,
            interpolation = self.interpolation,
        ) if target is not None else None
        return input, target

# endregion


# region Rotate

@constant.TRANSFORM.register(name="rotate")
class Rotate(coreml.Transform):
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.rotate(
            image         = input,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.rotate(
            image         = target,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="rotate_hflip")
@constant.TRANSFORM.register(name="rotate_horizontal_flip")
class RotateHorizontalFlip(coreml.Transform):

    def __init__(
        self,
        angle        : float,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.rotate_horizontal_flip(
            image         = input,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.rotate_horizontal_flip(
            image         = target,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="rotate_vflip")
@constant.TRANSFORM.register(name="rotate_vertical_flip")
class RotateVerticalFlip(coreml.Transform):
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.rotate_vertical_flip(
            image         = input,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.rotate_vertical_flip(
            image         = target,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target

# endregion


# region Shear

@constant.TRANSFORM.register(name="hshear")
@constant.TRANSFORM.register(name="horizontal_shear")
class HorizontalShear(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.shear_horizontal(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.shear_horizontal(
            image         = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="shear")
class Shear(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : list[float],
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.shear(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.shear(
            image         = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="yshear")
@constant.TRANSFORM.register(name="vertical_shear")
class VerticalShear(coreml.Transform):

    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.shear_vertical(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.shear_vertical(
            image         = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target
        
# endregion


# region Translate

@constant.TRANSFORM.register(name="htranslate")
@constant.TRANSFORM.register(name="horizontal_translate")
class HorizontalTranslate(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.translate_horizontal(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.translate_horizontal(
            image         = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target
      

@constant.TRANSFORM.register(name="htranslate_image_box")
@constant.TRANSFORM.register(name="horizontal_translate_image_box")
class HorizontalTranslateImageBox(coreml.Transform):

    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input, target = ci.translate_image_bbox_horizontal(
            image         = input,
            bbox= target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        return input, target


@constant.TRANSFORM.register(name="translate")
class Translate(coreml.Transform):

    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.translate(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.translate(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target


@constant.TRANSFORM.register(name="translate_image_box")
class TranslateImageBox(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input, target = ci.translate_image_bbox(
            image         = input,
            bbox= target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        return input, target
    

@constant.TRANSFORM.register(name="vtranslate")
@constant.TRANSFORM.register(name="vertical_translate")
class VerticalTranslate(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints  | None          = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        p            : float | None          = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input = ci.translate_vertical(
            image         = input,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        target = ci.translate_vertical(
            image         = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        ) if target is not None else None
        return input, target
    

@constant.TRANSFORM.register(name="vtranslate_image_box")
@constant.TRANSFORM.register(name="vertical_translate_image_box")
class VerticalTranslateImageBox(coreml.Transform):
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None           = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool                  = True,
        fill         : Floats                = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool                  = False,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input, target = ci.translate_image_bbox_vertical(
            image         = input,
            bbox= target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
        )
        return input, target

# endregion
