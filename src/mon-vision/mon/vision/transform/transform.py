#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.transform` module implements image transformations.
"""

from __future__ import annotations

__all__ = [
    "ToImage",
]

import math
from typing import Any

import numpy as np
import PIL.Image
import torch

from mon import coreimage as ci, coreml
from mon.vision.typing import (
    Float3T, Floats, Image, InterpolationModeType, Ints, PaddingModeType,
)


# region Affine

@coreml.TRANSFORM.register(name="affine")
class Affine(coreml.Transform):
    """:class:`Affine` applies affine transformation on the given images
    keeping image center invariant.
    
    Args:
        angle: Rotation angle in degrees between -180 and 180, clockwise
            direction.
        translate: Horizontal and vertical translations (post-rotation
            translation).
        scale: Overall scale.
        shear: Shear angle value in degrees between -180 to 180, clockwise
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
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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

@coreml.TRANSFORM.register(name="rgb_to_bgr")
class RGBToBGR(coreml.Transform):
    """:class:`RGBToBGR` converts the given images from RGB to BGR.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="rgb_to_grayscale")
class RGBToGrayscale(coreml.Transform):
    """:class:`RGBToGrayscale` converts the given images from RGB to grayscale.
    
    Args:
        rgb_weights: Weights that will be applied on each channel (RGB). Sum of
            the weights should add up to 1.0 ([0.299, 0.587, 0.114] or 255
            ([76, 150, 29]). Defaults to None.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="rgb_to_hsv")
class RGBToHSV(coreml.Transform):
    """:class:`RGBToHSV` converts the given images from RGB to HSV.
    
    Args:
        eps: Scalar to enforce numerical stability. Defaults to 1e-8.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="rgb_to_lab")
class RGBToLab(coreml.Transform):
    """:class:`RGBToLab` converts the given images from RGB to Lab.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="rgb_to_linear_rgb")
class RGBToLinearRGB(coreml.Transform):
    """:class:`RGBToLinearRGB` converts the given images from RGB to Linear
    RGB.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="rgb_to_luv")
class RGBToLUV(coreml.Transform):
    """:class:`RGBToLUV` converts the given images from RGB to LUV.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="rgb_to_rgba")
class RGBToRGBA(coreml.Transform):
    """:class:`RGBToRGBA` converts the given images from RGB to RGBA.

    Args:
        alpha_val: A float number or tensor for the alpha value.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="rgb_to_xyz")
class RGBToXYZ(coreml.Transform):
    """:class:`RGBToXyz` converts the given images from RGB to XYZ.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
    

@coreml.TRANSFORM.register(name="rgb_to_ycrcb")
class RGBToYCrCb(coreml.Transform):
    """:class:`RGBToYCrCb` converts the given images from RGB to YCrCb.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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


@coreml.TRANSFORM.register(name="rgb_to_yuv")
class RGBToYUV(coreml.Transform):
    """:class:`RGBToYUV` converts the given images from RGB to YUV.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
        
        
@coreml.TRANSFORM.register(name="rgb_to_yuv420")
class RGBToYUV420(coreml.Transform):
    """:class:`RGBToYUV420` converts the given images from RGB to YUV420.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
        

@coreml.TRANSFORM.register(name="rgb_to_yuv422")
class RGBToYUV422(coreml.Transform):
    """:class:`RGBToYUV422` converts the given images from RGB to YUV422.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
    

@coreml.TRANSFORM.register(name="rgba_to_bgr")
class RGBAToBGR(coreml.Transform):
    """:class:`RGBAToBGR` converts the given images from RGBA to BGR.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
    

@coreml.TRANSFORM.register(name="rgba_to_rgb")
class RGBaToRGB(coreml.Transform):
    """:class:`RGBAToRGB` converts the given images from RGBA to RGB.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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

@coreml.TRANSFORM.register(name="denormalize")
class Denormalize(coreml.Transform):
    """:Class:`Denormalize` denormalizes the given images.
    
    Args:
        min: Current minimum pixel value of the image. Defaults to 0.0.
        max: Current maximum pixel value of the image. Defaults to 255.0.
        new_min: New minimum pixel value of the image. Defaults to 0.0.
        new_max: New minimum pixel value of the image. Defaults to 1.0.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

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
    

@coreml.TRANSFORM.register(name="normalize")
class Normalize(coreml.Transform):
    """:class:`Normalize` normalizes the given images.
 
    Args:
        min: Current minimum pixel value of the image. Defaults to 0.0.
        max: Current maximum pixel value of the image. Defaults to 255.0.
        new_min: New minimum pixel value of the image. Defaults to 0.0.
        new_max: New minimum pixel value of the image. Defaults to 1.0.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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


@coreml.TRANSFORM.register(name="to_image")
class ToImage(coreml.Transform):
    """:class:`RGBAToRGB` converts the given images from
    :class:`torch.torch.Tensor` to :class:`np.ndarray`.

    Args:
        keepdim: If True, the function will keep the dimensions of the input
            tensor. Defaults to True.
        denormalize: If True, the image will be denormalized to [0, 255].
            Defaults to False.
    """
    
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


@coreml.TRANSFORM.register(name="to_tensor")
class ToTensor(coreml.Transform):
    """:class:`RGBAToRGB` converts the given images from :class:`PIL.Image` or
    :class:`np.ndarray` to :class:`torch.torch.Tensor`. Optionally, convert
    :param:`image` to channel-first format and normalize it.
    
    Args:
        keepdim: If True, the channel dimension will be kept. If False unsqueeze
            the image to match the shape [..., C, H, W]. Defaults to True
        normalize: If True, normalize the image to [0, 1]. Defaults to False
    """
    
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

@coreml.TRANSFORM.register(name="center_crop")
class CenterCrop(coreml.Transform):
    """:class:`CenterCrop` crops the given images at the center. If image size
    is smaller than output size along any edge, image is padded with 0 and then
    center cropped.

    Args:
        output_size: Desired output size of the crop. If size is an int instead
            of sequence like (h, w), a square crop (size, size) is made. If
            provided a sequence of length 1, it will be interpreted as (size[0],
            size[0]).
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    
    
@coreml.TRANSFORM.register(name="crop")
class Crop(coreml.Transform):
    """:class:`Crop` crops the given images at specified location and output
    size.
    
    Args:
        top: Vertical component of the top left corner of the crop box.
        left: Horizontal component of the top left corner of the crop box.
        height: Height of the crop box.
        width: Width of the crop box.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="five_crop")
class FiveCrop(coreml.Transform):
    """:class:`FiveCrop` crops the given images into four corners and the
    central crop.

    Args:
        size: Desired output size of the crop. If size is an int instead of
            sequence like (h, w), a square crop (size, size) is made. If
            provided a sequence of length 1, it will be interpreted as (size[0],
            size[0]).
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="ten_crop")
class TenCrop(coreml.Transform):
    """:class:`TenCrop` generates ten cropped images from the given image. Crop
    the given image into four corners and the central crop plus the flipped
    version of these (horizontal flipping is used by default).
   
    Notes:
        This transform returns a tuple of images and there may be a mismatch
        in the number of inputs and targets your `Dataset` returns.

    Args:
        size: Desired output size of the crop. If size is an int instead of
            sequence like (h, w), a square crop (size, size) is made. If
            provided a sequence of length 1, it will be interpreted as (size[0],
            size[0]).
        vflip: Use vertical flipping instead of horizontal. Defaults to False.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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

@coreml.TRANSFORM.register(name="hflip")
@coreml.TRANSFORM.register(name="horizontal_flip")
class HorizontalFlip(coreml.Transform):
    """:class:`HorizontalFlip` flips the given images horizontally.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.horizontal_flip(image=input)
        target = ci.horizontal_flip(image=target) \
            if target is not None else None
        return input, target
    

@coreml.TRANSFORM.register(name="hflip_image_box")
@coreml.TRANSFORM.register(name="horizontal_flip_image_box")
class HorizontalFlipImageBox(coreml.Transform):
    """:class:`HorizontalFlipImageBox` flips an image and a bounding box
    horizontally.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.horizontal_flip_image_box(image=input, box=target)


@coreml.TRANSFORM.register(name="vflip")
@coreml.TRANSFORM.register(name="vertical_flip")
class VerticalFlip(coreml.Transform):
    """:class:`VerticalFlip` flips the given images vertically.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
      
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input  = ci.vertical_flip(image=input)
        target = ci.vertical_flip(image=target) if target is not None else None
        return input, target
    
    
@coreml.TRANSFORM.register(name="vflip_image_box")
@coreml.TRANSFORM.register(name="vertical_flip_image_box")
class VerticalFlipImageBox(coreml.Transform):
    """:class:`VerticalFlipImageBox` flips an image and a bounding box
    vertically.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.vertical_flip_image_box(image=input, box=target)

# endregion


# region Intensity

@coreml.TRANSFORM.register(name="adjust_brightness")
class AdjustBrightness(coreml.Transform):
    """:class:`AdjustBrightness` adjusts brightness of the given images.

    Args:
        brightness_factor: How much to adjust the brightness. Can be any
            non-negative number. 0 gives a black image, 1 gives the original
            image while 2 increases the brightness by a factor of 2.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="adjust_contrast")
class AdjustContrast(coreml.Transform):
    """:class:`AdjustContrast` adjusts the contrast of the given images.

    Args:
        contrast_factor: How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
        p: Probability of the image being adjusted. Defaults to None means
        process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="adjust_gamma")
class AdjustGamma(coreml.Transform):
    """:class:`AdjustGamma` adjusts the gamma of the given images.

    Args:
        gamma: How much to adjust the gamma. Can be any non-negative number. 0
            gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        gain: Default to 1.0.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="adjust_hue")
class AdjustHue(coreml.Transform):
    """:class:`AdjustHue` adjusts thr hue of the given images.

    Args:
        hue_factor: How much to shift the hue channel. Should be in [-0.5, 0.5].
            0.5 and -0.5 give complete reversal of hue channel in HSV space in
            positive and negative direction respectively. 0 means no shift.
            Therefore, both -0.5 and 0.5 will give an image with complementary
            colors while 0 gives the original image.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="adjust_saturation")
class AdjustSaturation(coreml.Transform):
    """:class:`AdjustSaturation` adjusts the color saturation of the given
    images.

    Args:
        saturation_factor: How much to adjust the saturation. 0 will give a
            black and white image, 1 will give the original image while 2 will
            enhance the saturation by a factor of 2.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="adjust_sharpness")
class AdjustSharpness(coreml.Transform):
    """:class:`AdjustSharpness` adjust the sharpness of the given images.

    Args:
        sharpness_factor: How much to adjust the sharpness. 0 will give a black
            and white image, 1 will give the original image while 2 will enhance
            the saturation by a factor of 2.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="autocontrast")
class AutoContrast(coreml.Transform):
    """:class:`AutoContrast` maximizes the contrast of the given images by
    remapping its pixels per channel so that the lowest becomes black and the
    lightest becomes white.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
    

@coreml.TRANSFORM.register(name="color_jitter")
class ColorJitter(coreml.Transform):
    """:class:`ColorJitter` randomly changes the brightness, contrast,
    saturation and hue of the given images.
    
    Args:
        brightness: How much to jitter the brightness. :param:`brightness` is
            chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or
            the given [min, max]. Should be non-negative numbers. Defaults to
            0.0.
        contrast: How much to jitter the contrast. :param:`contrast` is chosen
            uniformly from [max(0, 1 - contrast), 1 + contrast] or the given
            [min, max]. Should be non-negative numbers. Defaults to 0.0.
        saturation: How much to jitter the saturation. :param:`saturation` is
            chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or
            the given [min, max]. Should be non-negative numbers. Defaults to
            0.0.
        hue: How much to jitter the hue. :param:`hue` is chosen uniformly from
            [-hue, hue] or the given [min, max]. Should have 0 <= hue <= 0.5 or
            -0.5 <= min <= max <= 0.5. Defaults to 0.0.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        """Gets the parameters for the randomized transform to be applied on
        image.

        Args:
            brightness: The range from which the `brightness_factor` is chosen
                uniformly. Pass None to turn off the transformation.
            contrast: The range from which the `contrast_factor` is chosen
                uniformly. Pass None to turn off the transformation.
            saturation: The range from which the `saturation_factor` is chosen
                uniformly. Pass None to turn off the transformation.
            hue: The range from which the `hue_factor` is chosen uniformly. Pass
                None to turn off the transformation.

        Returns:
            The parameters used to apply the randomized transform along with
                their random order.
        """
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


@coreml.TRANSFORM.register(name="erase")
class Erase(coreml.Transform):
    """:class:`Erase` erases value of the given images.

    Args:
        image: Image of shape [..., C, H, W] to be adjusted, where ... means it
            can have an arbitrary number of leading dimensions.
        i: i in (i,j) i.e coordinates of the upper left corner.
        j: j in (i,j) i.e coordinates of the upper left corner.
        h: Height of the erased region.
        w: Width of the erased region.
        v: Erasing value.
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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


@coreml.TRANSFORM.register(name="equalize")
class Equalize(coreml.Transform):
    """:class:`Equalize` equalizes histograms of the given images by applying a
    non-linear mapping to the input in order to create a uniform distribution of
    grayscale values in the output.
    
    Args:
        p: Probability of the image being adjusted. Defaults to None means
            process as normal.
    """
    
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
        

@coreml.TRANSFORM.register(name="invert")
class Invert(coreml.Transform):
    """
    Invert the colors of an RGB/grayscale image.
    
    Args:
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """

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


@coreml.TRANSFORM.register(name="posterize")
class Posterize(coreml.Transform):
    """
    Posterize an image by reducing the number of bits for each color channel.

    Args:
        bits: Number of bits to keep for each channel (0-8).
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
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


@coreml.TRANSFORM.register(name="random_erase")
class RandomErase(coreml.Transform):
    """
    Randomly selects a rectangle region in an image torch.Tensor and erases its
    pixels.
    
    References:
        'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    
    Args:
        scale: Range of proportion of erased area against input image.
        ratio: Range of aspect ratio of erased area.
        value (int | float | str | tuple | list): Erasing value. Defaults to 0.
            If a single int, it is used to erase all pixels. If a tuple of
            length 3, it is used to erase R, G, B channels respectively. If a
            str of `random`, erasing each pixel with random values.
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        scale  : Floats                           = (0.02, 0.33),
        ratio  : Floats                           = (0.3, 3.3),
        value  : int | float | str | tuple | list = 0,
        inplace: bool                             = False,
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
        self.inplace = inplace
        
    @staticmethod
    def get_params(
        image: torch.Tensor,
        scale: Floats,
        ratio: Floats,
        value: list[float] | None = None
    ) -> tuple[int, int, int, int, torch.Tensor]:
        """Get parameters for `erase` for a random erasing.

        Args:
            image (torch.Tensor): torch.Tensor image to be erased.
            scale: Range of proportion of erased area against input
                image.
            ratio: Range of aspect ratio of erased area.
            value (list[float] | None): Erasing value. If None, it is
                interpreted as "random" (erasing each pixel with random values).
                If `len(value)` is 1, it is interpreted as a number, i.e.
                `value[0]`.

        Returns:
            Params (i, j, h, w, v) to be passed to `erase` for random erasing.
        """
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
                inplace = self.inplace,
            ), \
            ci.erase(
                image   = target,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
                inplace = self.inplace,
            ) if target is not None else None
    

@coreml.TRANSFORM.register(name="solarize")
class Solarize(coreml.Transform):
    """
    Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        threshold: All pixels equal or above this value are inverted.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
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
        return ci.solarize(image=input,  threshold=self.threshold), \
               ci.solarize(image=target, threshold=self.threshold) \
                   if target is not None else None

# endregion


# region Resize

@coreml.TRANSFORM.register(name="resize")
class Resize(coreml.Transform):
    """
    Resize an image. Adapted from: `torchvision.transforms.functional.resize()`
    
    Args:
        image (torch.Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        size: Desired output size of shape [C, H, W].
        interpolation (InterpolationModeType): Interpolation method.
        antialias (bool, None): Defaults to None.
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        size         : Ints,
        interpolation: InterpolationModeType = "bilinear",
        antialias    : bool  | None       = None,
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.size          = size
        self.interpolation = interpolation
        self.antialias     = antialias
        self.inplace       = inplace
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.resize(
                image         = input,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
                inplace       = self.inplace,
            ), \
            ci.resize(
                image         = target,
                size          = self.size,
                interpolation = self.interpolation,
                antialias     = self.antialias,
                inplace       = self.inplace,
            ) if target is not None else None
    

@coreml.TRANSFORM.register(name="resized_crop")
class ResizedCrop(coreml.Transform):
    """
    Crop and resize an image.
    
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        top: Vertical component of the top left corner of the crop box.
        left: Horizontal component of the top left corner of the crop box.
        height: Height of the crop box.
        width: Width of the crop box.
        size: Desired output size of shape [C, H, W].
            Defaults to None.
        interpolation (InterpolationModeType): Interpolation method.
            Defaults to "bilinear".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: InterpolationModeType = "bilinear",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation
        self.inplace       = inplace
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.resized_crop(
                image= input,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation,
                inplace       = self.inplace,
            ), \
            ci.resized_crop(
                image= target,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation,
                inplace       = self.inplace
            ) if target is not None else None

# endregion


# region Rotate

@coreml.TRANSFORM.register(name="rotate")
class Rotate(coreml.Transform):
    """
    Rotate an image.
    
    Args:
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Defaults to True.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation.
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None  = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.rotate(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.rotate(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@coreml.TRANSFORM.register(name="rotate_hflip")
@coreml.TRANSFORM.register(name="rotate_horizontal_flip")
class RotateHorizontalFlip(coreml.Transform):
    """
    Rotate an image, then flips it horizontally.
    
    Args:
        angle: Angle to rotate the image.
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None  = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.rotate_horizontal_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.rotate_horizontal_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
    

@coreml.TRANSFORM.register(name="rotate_vflip")
@coreml.TRANSFORM.register(name="rotate_vertical_flip")
class RotateVerticalFlip(coreml.Transform):
    """
    Rotate an image, then flips it vertically.
    
    Args:
        image (torch.Tensor): Image of shape [..., C, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        angle: Angle to rotate the image.
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        angle        : float,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.rotate_vertical_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.rotate_vertical_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None

# endregion


# region Shear

@coreml.TRANSFORM.register(name="hshear")
@coreml.TRANSFORM.register(name="horizontal_shear")
class HorizontalShear(coreml.Transform):
    """
    Shear an image horizontally.
    
    Args:
        magnitude: Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode. Defaults to
            "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.horizontal_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.horizontal_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
    

@coreml.TRANSFORM.register(name="shear")
class Shear(coreml.Transform):
    """
    Shear an image.
    
    Args:
        magnitude: Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
            translation. Defaults to True
        fill: Pixel values for the area outside the transformed image.
            - If a single number, the value is used for all borders.
            - If a sequence of length 2, it is used to fill band left/right and
              top/bottom respectively
            - If a sequence of length 3, it is used to fill R, G, B channels
              respectively.
            - If a sequence of length 4, it is used to fill each band
              (left, right, top, bottom) respectively.
            Defaults to 0.0.
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : list[float],
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@coreml.TRANSFORM.register(name="yshear")
@coreml.TRANSFORM.register(name="vertical_shear")
class VerticalShear(coreml.Transform):
    """
    Shear an image vertically.
    
    Args:
        magnitude: Shear angle value in degrees between -180 to 180,
            clockwise direction.
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to  make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : float,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.vertical_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.vertical_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None

# endregion


# region Translate

@coreml.TRANSFORM.register(name="htranslate")
@coreml.TRANSFORM.register(name="horizontal_translate")
class HorizontalTranslate(coreml.Transform):
    """
    Translate an image in horizontal direction.
    
    Args:
        magnitude: Horizontal translation (post-rotation translation)
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode. Defaults to
            "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.horizontal_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.horizontal_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None
      

@coreml.TRANSFORM.register(name="htranslate_image_box")
@coreml.TRANSFORM.register(name="horizontal_translate_image_box")
class HorizontalTranslateImageBox(coreml.Transform):
    """
    Translate an image and a bounding box in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        magnitude: Horizontal translation (post-rotation translation).
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode. Defaults to
            "constant".
        drop_ratio: If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.horizontal_translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )


@coreml.TRANSFORM.register(name="translate")
class Translate(coreml.Transform):
    """
    Translate an image.
    
    Args:
        magnitude: Horizontal and vertical translations (post-rotation
            translation).
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@coreml.TRANSFORM.register(name="translate_image_box")
class TranslateImageBox(coreml.Transform):
    """
    Translate an image and a bounding box.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        box (torch.Tensor): Box of shape [N, 4] to be translated. They are expected
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        magnitude: Horizontal and vertical translations (post-rotation
            translation).
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        drop_ratio: If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : Ints,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )
    

@coreml.TRANSFORM.register(name="vtranslate")
@coreml.TRANSFORM.register(name="vertical_translate")
class VerticalTranslate(coreml.Transform):
    """
    Translate an image in vertical direction.
    
    Args:
        magnitude: Vertical translation (post-rotation translation)
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None  = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
        
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return \
            ci.vertical_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ), \
            ci.vertical_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                fill          = self.fill,
                padding_mode  = self.padding_mode,
                inplace       = self.inplace,
            ) if target is not None else None


@coreml.TRANSFORM.register(name="vtranslate_image_box")
@coreml.TRANSFORM.register(name="vertical_translate_image_box")
class VerticalTranslateImageBox(coreml.Transform):
    """
    Translate an image and a bounding box in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        magnitude: Vertical translation (post-rotation translation).
        center: Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        interpolation (InterpolationModeType): Desired interpolation mode.
            Default to "bilinear".
        keep_shape: If True, expands the output image to make it large
            enough to hold the entire rotated image. If False or omitted, make
            the output image the same size as the input image. Note that the
            `keep_shape` flag assumes rotation around the center and no
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
        padding_mode (PaddingModeType): Desired padding mode.
            Defaults to "constant".
        drop_ratio: If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace: If True, make this operation inplace. Defaults to False.
        p: Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        magnitude    : int,
        center       : Ints | None        = None,
        interpolation: InterpolationModeType = "bilinear",
        keep_shape   : bool               = True,
        fill         : Floats             = 0.0,
        padding_mode : PaddingModeType       = "constant",
        inplace      : bool               = False,
        p            : float | None       = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.fill          = fill
        self.padding_mode  = padding_mode
        self.inplace       = inplace
    
    # noinspection PyMethodOverriding
    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ci.vertical_translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            fill          = self.fill,
            padding_mode  = self.padding_mode,
            inplace       = self.inplace,
        )

# endregion
