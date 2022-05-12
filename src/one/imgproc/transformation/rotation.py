#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional
from typing import Union

import numpy as np
from torch import nn
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip

from one.core import FloatAnyT
from one.core import ListOrTuple2T
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.transformation.affine import affine
from one.imgproc.transformation.affine import affine_image_box

__all__ = [
    "rotate",
    "rotate_image_box",
    "rotate_hflip",
    "rotate_vflip",
    "Rotate",
    "RotateHflip",
    "RotateVflip",
]


# MARK: - Functional

def rotate(
    image        : TensorOrArray,
    angle        : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> Tensor:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
    
    Args:
        image (TensorOrArray[*, C, H, W]):
            Image to be rotated.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image. Default: `True`.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (TensorOrArray[*, C, H, W]):
            Rotated image.
    """
    if isinstance(image, Tensor):
        return affine(
            image         = image,
            angle         = angle,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [0, 0],
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
            pad_mode      = pad_mode,
            fill          = fill,
        )
    elif isinstance(image, np.ndarray):
        return affine(
            image         = image,
            angle         = angle,
            translate     = [0, 0],
            scale         = 1.0,
            shear         = [0, 0],
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
            pad_mode      = pad_mode,
            fill          = fill,
        )
    else:
        raise ValueError(f"Do not support {type(image)}.")


def rotate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    angle        : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
    drop_ratio   : float                        = 0.0
) -> tuple[TensorOrArray, TensorOrArray]:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
    
    Args:
        image (TensorOrArray[*, C, H, W]):
            Image to be rotated.
        box (TensorOrArray[B, 4]):
            Bounding boxes to be rotated.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
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
        image (TensorOrArray[*, C, H, W]):
            Rotated image.
        box (TensorOrArray[B, 4]):
            Rotated bounding boxes.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    
    return affine_image_box(
        image         = image,
        box           = box,
        angle         = angle,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
        drop_ratio    = drop_ratio,
    )


def rotate_hflip(
    image        : TensorOrArray,
    angle        : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Rotate a tensor image or a batch of tensor images and then horizontally
    flip.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (TensorOrArray[B, C, H, W]):
            Rotated and flipped image.
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )
    return hflip(image)


def rotate_vflip(
    image        : TensorOrArray,
    angle        : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Rotate a tensor image or a batch of tensor images and then vertically
    flip.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (TensorOrArray[B, C, H, W]):
            Rotated and flipped image.
    """
    image = rotate(
        image         = image,
        angle         = angle,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )
    return vflip(image)


# MARK: - Modules

@TRANSFORMS.register(name="rotate")
class Rotate(nn.Module):
    """Rotate a tensor image or a batch of tensor images.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to be rotated.
                
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Rotated image.
        """
        return rotate(
            image         = image,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="rotate_hflip")
class RotateHflip(nn.Module):
    """Rotate a tensor image or a batch of tensor images and then horizontally
    flip.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to be rotated and horizontally flipped.
                
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Rotated and horizontally flipped image.
        """
        return rotate_hflip(
            image         = image,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="rotate_vflip")
class RotateVflip(nn.Module):
    """Rotate a tensor image or a batch of tensor images and then vertically
    flip.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.NEAREST`,
            `InterpolationMode.BILINEAR` are supported. For backward
            compatibility integer values (e.g. `PIL.Image.NEAREST`) are still
            acceptable.
        keep_shape (bool):
            If `True`, expands the output image to  make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, optional):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to be rotated and vertically flipped.
                
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Rotated and vertically flipped image.
        """
        return rotate_vflip(
            image         = image,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )
