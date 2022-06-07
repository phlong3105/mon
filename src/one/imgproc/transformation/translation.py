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
from one.core import Int2T
from one.core import ListOrTuple2T
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.transformation.affine import affine
from one.imgproc.transformation.affine import affine_image_box

__all__ = [
    "translate",
    "translate_image_box",
    "htranslate",
    "htranslate_image_box",
    "vtranslate",
    "vtranslate_image_box",
    "Translate",
    "TranslateImageBox",
    "Htranslate",
    "HtranslateImageBox",
    "Vtranslate",
    "VtranslateImageBox",
]


# MARK: - Functional

def translate(
    image        : TensorOrArray,
    magnitude    : Int2T,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Translate image in vertical and horizontal direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (Int2T):
            Horizontal and vertical translations (post-rotation translation).
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
        translate     = magnitude,
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


def translate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
	magnitude    : Int2T,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
    drop_ratio   : float                        = 0.0
) -> tuple[TensorOrArray, TensorOrArray]:
    """Translate the image and bounding box with the given magnitude.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be translated.
        box (TensorOrArray[B, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (Int2T):
            Horizontal and vertical translations (post-rotation translation).
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
        image (TensorOrArray[B, C, H, W]):
            Translated image with the shape as the specified size.
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")

    return affine_image_box(
	    image         = image,
		box           = box,
	    angle         = 0.0,
		translate     = magnitude,
	    scale         = 1.0,
	    shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
		drop_ratio    = drop_ratio,
    )


def htranslate(
    image        : TensorOrArray,
    magnitude    : int,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Translate image in horizontal direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Horizontal translation (post-rotation translation)
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
        translate     = [magnitude, 0],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


def htranslate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
	magnitude    : int,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
    drop_ratio   : float                        = 0.0
) -> tuple[TensorOrArray, TensorOrArray]:
    """Translate the image and bounding box in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be translated.
        box (TensorOrArray[B, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (int):
            Horizontal translation (post-rotation translation).
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
        image (TensorOrArray[B, C, H, W]):
            Translated image with the shape as the specified size.
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
   
    return affine_image_box(
	    image         = image,
		box           = box,
	    angle         = 0.0,
		translate     = [magnitude, 0],
	    scale         = 1.0,
	    shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
		drop_ratio    = drop_ratio,
    )


def vtranslate(
    image        : TensorOrArray,
    magnitude    : int,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Translate image in vertical direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Vertical translation (post-rotation translation)
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
        translate     = [0, magnitude],
        scale         = 1.0,
        shear         = [0.0, 0.0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
    )


def vtranslate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
	magnitude    : int,
    center       : Optional[ListOrTuple2T[int]] = None,
	interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
	keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
	fill         : Optional[FloatAnyT]          = None,
    drop_ratio   : float                        = 0.0
) -> tuple[TensorOrArray, TensorOrArray]:
    """Translate the image and bounding box in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be translated.
        box (TensorOrArray[B, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (int):
            Vertical translation (post-rotation translation).
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
        image (TensorOrArray[B, C, H, W]):
            Translated image with the shape as the specified size.
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
   
    return affine_image_box(
	    image         = image,
		box           = box,
	    angle         = 0.0,
		translate     = [0, magnitude],
	    scale         = 1.0,
	    shear         = [0, 0],
        center        = center,
        interpolation = interpolation,
        keep_shape    = keep_shape,
        pad_mode      = pad_mode,
        fill          = fill,
		drop_ratio    = drop_ratio,
    )


# MARK: - Modules

@TRANSFORMS.register(name="translate")
class Translate(nn.Module):
    """
    
    Args:
        magnitude (Int2T):
            Horizontal and vertical translation magnitude.
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
        magnitude    : Int2T,
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
        return translate(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="translate_image_box")
class TranslateImageBox(nn.Module):
    """
    
    Args:
        magnitude (Int2T):
            Horizontal and vertical translation magnitude.
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
        magnitude    : Int2T,
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
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to transform.
            box (TensorOrArray[B, 4]):
                Box to be translated. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
            
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
            box (TensorOrArray[B, 4]):
                Translated boxes.
        """
        return translate_image_box(
            image         = image,
            box           = box,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="htranslate")
@TRANSFORMS.register(name="horizontal_translate")
class Htranslate(nn.Module):
    """
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
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
        magnitude    : int,
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
        return htranslate(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="htranslate_image_box")
@TRANSFORMS.register(name="horizontal_translate_image_box")
class HtranslateImageBox(nn.Module):
    """
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
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
        magnitude    : int,
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
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to transform.
            box (TensorOrArray[B, 4]):
                Box to be translated. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
            
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
            box (TensorOrArray[B, 4]):
                Translated boxes.
        """
        return htranslate_image_box(
            image         = image,
            box           = box,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="vtranslate")
@TRANSFORMS.register(name="vertical_translate")
class Vtranslate(nn.Module):
    """
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
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
        magnitude    : int,
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
        return vtranslate(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="vtranslate_image_box")
@TRANSFORMS.register(name="vertical_translate_image_box")
class VtranslateImageBox(nn.Module):
    """
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
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
        magnitude    : int,
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
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        """
        
        Args:
            image (TensorOrArray[B, C, H, W]):
                Image to transform.
            box (TensorOrArray[B, 4]):
                Box to be translated. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
            
        Returns:
            image (TensorOrArray[B, C, H, W]):
                Transformed image.
            box (TensorOrArray[B, 4]):
                Translated boxes.
        """
        return vtranslate_image_box(
            image         = image,
            box           = box,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )
