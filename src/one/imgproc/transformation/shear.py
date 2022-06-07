#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional
from typing import Union

from torch import nn
from torchvision.transforms import InterpolationMode

from one.core import FloatAnyT
from one.core import ListOrTuple2T
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.transformation.affine import affine
from one.imgproc.transformation.affine import affine_image_box

__all__ = [
    "shear",
    "shear_image_box",
    "hshear",
    "vshear",
    "Shear",
    "Hshear",
    "Vshear",
]


# MARK: - Functional

def shear(
    image        : TensorOrArray,
    magnitude    : FloatAnyT,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Shear image.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x axis, while the second value
            corresponds to a shear parallel to the y axis.
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
            Transformed image.
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


def shear_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    magnitude    : FloatAnyT,
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
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x axis, while the second value
            corresponds to a shear parallel to the y axis.
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
	    angle         = 0.0,
		translate     = [0, 0],
	    scale         = 1.0,
	    shear         = magnitude,
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
		drop_ratio    = drop_ratio,
    )


def hshear(
    image        : TensorOrArray,
    magnitude    : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Shear image horizontally.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
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
            Transformed image.
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [magnitude, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


def vshear(
    image        : TensorOrArray,
    magnitude    : float,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Shear image vertically.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
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
            Transformed image.
    """
    return affine(
        image         = image,
        angle         = 0.0,
        translate     = [0, 0],
        scale         = 1.0,
        shear         = [0.0, magnitude],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


# MARK: - Modules

@TRANSFORMS.register(name="shear")
class Shear(nn.Module):
    """
    
    Args:
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x axis, while the second value
            corresponds to a shear parallel to the y axis.
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
        magnitude    : list[float],
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.magnitude     = magnitude
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
                Image to transform.

        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
        """
        return shear(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill
        )


@TRANSFORMS.register(name="hshear")
@TRANSFORMS.register(name="horizontal_shear")
class Hshear(nn.Module):
    """
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
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
        magnitude    : float,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.magnitude     = magnitude
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
                Image to transform.
 
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
        """
        return hshear(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill
        )


@TRANSFORMS.register(name="yshear")
@TRANSFORMS.register(name="vertical_shear")
class Vshear(nn.Module):
    """
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
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
        magnitude    : float,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.magnitude     = magnitude
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
                Image to transform.

        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
        """
        return vshear(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill
        )
