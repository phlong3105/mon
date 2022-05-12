#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import math
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import functional as F

from one.core import Float2T
from one.core import FloatAnyT
from one.core import get_image_size
from one.core import InterpolationMode
from one.core import pad_image
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.spatial.box import scale_box
from one.imgproc.transformation.resize import resize

__all__ = [
    "padded_scale",
    "scale",
    "scale_image_box",
    "PaddedScale",
    "Scale",
    "ScaleImageBox",
]


# MARK: - Functional

def padded_scale(image: Tensor, ratio: float = 1.0, same_shape: bool = False) -> Tensor:
    """Scale image with the ratio and pad the border.
    
    Args:
        image (Tensor):
            Input image.
        ratio (float):
            Ratio to scale the image (mostly scale down).
        same_shape (bool):
            If `True`, pad the scaled image to retain the original [H, W].
            
    Returns:
        scaled_image (Tensor):
            Scaled image.
    """
    # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return image
    else:
        h, w = image.shape[2:]
        s    = (int(h * ratio), int(w * ratio))  # new size
        img  = F.interpolate(image, size=s, mode="bilinear", align_corners=False)  # Resize
        if not same_shape:  # Pad/crop img
            gs   = 128  # 64 # 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
    
    
def scale(
    image        : TensorOrArray,
    factor       : Float2T,
    interpolation: InterpolationMode       = InterpolationMode.BILINEAR,
    antialias    : bool                    = False,
    keep_shape   : bool                    = False,
    pad_mode     : Union[PaddingMode, str] = "constant",
    fill         : Optional[FloatAnyT]     = None,
) -> TensorOrArray:
    """Scale the image with the given factor. Optionally, pad the scaled up
    image

    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be scaled.
        factor (Float2T):
            Desired scaling factor in each direction. If scalar, the value is
            used for both the vertical and horizontal direction.
            If factor > 1.0, scale up. If factor < 1.0, scale down.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        antialias (bool, optional):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.
        keep_shape (bool):
            When `True`, pad the scaled image with `fill` to retain the original
            [H, W] if scaling down. Default: `False`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (TensorOrArray[B, C, H * factor, W * factor]):
            Rescaled image with the shape as the specified size.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if pad_mode not in ("constant", PaddingMode.CONSTANT):
        raise ValueError(f"Current only support pad_mode == 'constant'."
                         f"But got: {pad_mode}.")
    if isinstance(factor, float):
        factor_ver = factor_hor = factor
    else:
        factor_ver, factor_hor  = factor
    if factor_ver <= 0 or factor_hor <= 0:
        raise ValueError(f"factor values must >= 0. But got: {factor}")
    
    h0, w0 = get_image_size(image)
    h1, w1 = int(h0 * factor_ver), int(w0 * factor_hor)
    scaled = resize(
        image         = image,
        size          = (h1, w1),
        interpolation = interpolation,
        antialias     = antialias
    )
    
    # NOTE: Pad to original [H, W]
    if keep_shape:
        return pad_image(
            image    = scaled,
            pad_size = (h0, w0),
            mode     = pad_mode,
            value    = fill,
        )
    
    return scaled


def scale_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    factor       : Float2T,
    interpolation: InterpolationMode       = InterpolationMode.BILINEAR,
    antialias    : bool                    = False,
    keep_shape   : bool                    = False,
    pad_mode     : Union[PaddingMode, str] = "constant",
    fill         : Optional[FloatAnyT]     = None,
    drop_ratio   : float                   = 0.0
) -> tuple[TensorOrArray, TensorOrArray]:
    """Scale the image and bounding box with the given factor.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be scaled.
        box (TensorOrArray[B, 4]):
            Box to be scaled. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        factor (Float2T):
            Desired scaling factor in each direction. If scalar, the value is
            used for both the vertical and horizontal direction.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        antialias (bool, optional):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.
        keep_shape (bool):
            When `True`, pad the scaled image with `fill` to retain the original
            [H, W] if scaling down. Default: `False`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (TensorOrArray[B, C, H, W]):
            Rescaled image with the shape as the specified size.
        box (TensorOrArray[B, 4]):
            Rescaled boxes.
         
    Example:
        >>> img = torch.rand(1, 3, 4, 4)
        >>> out = scale(img, (2, 3))
        >>> print(out.shape)
        torch.Size([1, 3, 8, 12])
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if pad_mode not in ("constant", PaddingMode.CONSTANT):
        raise ValueError(f"Current only support pad_mode == 'constant'."
                         f"But got: {pad_mode}.")
    
    image_size = get_image_size(image)
    return \
	    scale(
            image         = image,
            factor        = factor,
            interpolation = interpolation,
            antialias     = antialias,
            keep_shape    = keep_shape,
            pad_mode      = pad_mode,
            fill          = fill,
        ), \
	    scale_box(
            box        = box,
            cur_size= image_size,
            factor     = factor,
            keep_shape = keep_shape,
            drop_ratio = drop_ratio,
        )


# MARK: - Modules

@TRANSFORMS.register(name="padded_scale")
class PaddedScale(nn.Module):
    
    def __init__(self, ratio: float = 1.0, same_shape: bool = False):
        super().__init__()
        self.ratio      = ratio
        self.same_shape = same_shape
    
    def forward(self, image: Tensor) -> Tensor:
        return padded_scale(image, self.ratio, self.same_shape)


@TRANSFORMS.register(name="scale")
class Scale(nn.Module):
    r"""Rescale the input image with the given factor.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Args:
        factor (Float2T):
            Desired scaling factor in each direction. If scalar, the value is
            used for both the vertical and horizontal direction.
            If factor > 1.0, scale up. If factor < 1.0, scale down.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        antialias (bool, optional):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.
        keep_shape (bool):
            When `True`, pad the scaled image with `fill` to retain the original
            [H, W] if scaling down. Default: `False`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    def __init__(
        self,
        factor       : Float2T,
        interpolation: InterpolationMode       = InterpolationMode.BILINEAR,
        antialias    : bool                    = False,
        keep_shape   : bool                    = False,
        pad_mode     : Union[PaddingMode, str] = "constant",
        fill         : Optional[FloatAnyT]     = None,
    ):
        super().__init__()
        self.factor        = factor
        self.interpolation = interpolation
        self.antialias     = antialias
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to be scaled.
        
        Returns:
            image(TensorOrArray[B, C, H, W]):
                Rescaled image.
        """
        return scale(
            image         = image,
            factor        = self.factor,
            interpolation = self.interpolation,
            antialias     = self.antialias,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="scale_image_box")
class ScaleImageBox(nn.Module):
    r"""Scale the image and bounding box with the given factor.
    
    Attributes:
        factor (Float2T):
            Desired scaling factor in each direction. If scalar, the value is
            used for both the vertical and horizontal direction.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        antialias (bool, optional):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.
        keep_shape (bool):
            When `True`, pad the scaled image with `fill` to retain the original
            [H, W] if scaling down. Default: `False`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
    """
    
    def __init__(
        self,
        factor       : Float2T,
        interpolation: InterpolationMode       = InterpolationMode.BILINEAR,
        antialias    : bool                    = False,
        keep_shape   : bool                    = False,
        pad_mode     : Union[PaddingMode, str] = "constant",
        fill         : Optional[FloatAnyT]     = None,
        drop_ratio   : float                   = 0.0
    ):
        super().__init__()
        self.factor        = factor
        self.interpolation = interpolation
        self.antialias     = antialias
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
        self.drop_ratio    = drop_ratio
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to be scaled.
            box (TensorOrArray[B, 4]):
                Box to be scaled. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
            
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Rescaled image.
            box (TensorOrArray[B, 4]):
                Rescaled boxes.
        """
        return scale_image_box(
            image         = image,
            box           = box,
            factor        = self.factor,
            interpolation = self.interpolation,
            antialias     = self.antialias,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
            drop_ratio    = self.drop_ratio
        )
