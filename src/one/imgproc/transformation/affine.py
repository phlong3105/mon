#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""An affine transformation is any transformation that preserves collinearity
(i.e., all points lying on a line initially still lie on a line after
transformation) and ratios of distances (e.g., the midpoint of a line segment
remains the midpoint after transformation). In this sense, affine indicates a
special class of projective transformations that do not move any objects from
the affine space R^3 to the plane at infinity or conversely. An affine
transformation is also called an affinity.

Geometric contraction, expansion, dilation, reflection, rotation, shear,
similarity transformations, spiral similarities, and translation are all
affine transformations, as are their combinations. In general, an affine
transformation is a composition of rotations, translations, dilations,
and shears.

While an affine transformation preserves proportions on lines, it does not
necessarily preserve angles or lengths. Any triangle can be transformed into
any other by an affine transformation, so all triangles are affine and,
in this sense, affine is a generalization of congruent and similar.
"""

from __future__ import annotations

import math
from typing import Optional
from typing import Union

import cv2
import numpy as np
from torch import nn
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import _get_inverse_affine_matrix

from one.core import FloatAnyT
from one.core import get_image_size
from one.core import IntAnyT
from one.core import InterpolationMode
from one.core import ListOrTuple2T
from one.core import pad_image
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.spatial import affine_box
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "affine",
    "affine_image_box",
    "Affine",
]


# MARK: - Functional

def _affine_tensor_image(
    image        : Tensor,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        image (Tensor[B, C, H, W]):
            Image to be transformed.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale.
        shear (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
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
        image (Tensor[B, C, H, W]):
            Transformed image.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(f"`angle` must be `int` or `float`. But got: {type(angle)}.")
    if isinstance(angle, int):
        angle = float(angle)
    
    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if isinstance(translate, tuple):
        translate = list(translate)
    if not isinstance(translate, (list, tuple)):
        raise TypeError(f"`translate` must be `list` or `tuple`. But got: {type(translate)}.")
    if len(translate) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(translate)}.")
    
    if isinstance(scale, int):
        scale = float(scale)
    if scale < 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")

    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(f"`shear` must be a single value or a sequence of length 2. But got: {shear}.")
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(shear)}.")
   
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    # if not isinstance(interpolation, InterpolationMode):
    #    raise TypeError(f"`interpolation` must be a `InterpolationMode`. But got: {type(interpolation)}.")

    img    = image.clone()
    h, w   = get_image_size(img)
    center = (h * 0.5, w * 0.5) if center is None else center  # H, W
    center = tuple(center[::-1])  # W, H
    
    if not isinstance(image, Tensor):
        # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        matrix            = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        pil_interpolation = InterpolationMode.PILModesMapping[interpolation]
        return F_pil.affine(image, matrix=matrix, interpolation=pil_interpolation, fill=fill)

    # If keep shape, find the new width and height bounds
    if not keep_shape:
        matrix  = _get_inverse_affine_matrix([0, 0], angle, [0, 0], 1.0, [0.0, 0.0])
        abs_cos = abs(matrix[0])
        abs_sin = abs(matrix[1])
        new_w   = int(h * abs_sin + w * abs_cos)
        new_h   = int(h * abs_cos + w * abs_sin)
        image   = pad_image(image, pad_size=(new_h, new_w))
    
    translate_f = [1.0 * t for t in translate]
    matrix      = _get_inverse_affine_matrix([0, 0], angle, translate_f, scale, shear)
    return F_t.affine(image, matrix=matrix, interpolation=interpolation.value, fill=fill)


@channel_last_processing
def _affine_numpy_image(
    image        : np.ndarray,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> np.ndarray:
    """Apply affine transformation on the image keeping image center invariant.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
        https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        
    Args:
        image (np.ndarray[C, H, W]):
            Image to be transformed. The image is converted to channel last
            format during processing.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale.
        shear (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
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
        image (np.ndarray[C, H, W]):
            Transformed image.
    """
    if image.ndim != 3:
        raise ValueError(f"`image.ndim` must be 3. But got: {image.ndim}.")
  
    if not isinstance(angle, (int, float)):
        raise TypeError(f"`angle` must be `int` or `float`. But got: {type(angle)}.")
    if isinstance(angle, int):
        angle = float(angle)
    
    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if not isinstance(translate, (list, tuple)):
        raise TypeError(f"`translate` must be `list` or `tuple`. But got: {type(translate)}.")
    if isinstance(translate, tuple):
        translate = list(translate)
    if len(translate) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(translate)}.")
    
    if isinstance(scale, int):
        scale = float(scale)
    if scale < 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")
   
    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(f"`shear` must be a single value or a sequence of length 2. But got: {shear}.")
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(shear)}.")
    
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError(f"`interpolation` must be a `InterpolationMode`. But got: {type(interpolation)}.")
    
    img    = image.copy()
    h, w   = get_image_size(img)
    center = (h * 0.5, w * 0.5) if center is None else center  # H, W
    center = tuple(center[::-1])  # W, H
    angle  = -angle
    R      = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)

    # If keep shape, find the new width and height bounds
    if keep_shape:
        new_w = w
        new_h = h
    else:
        abs_cos  = abs(R[0, 0])
        abs_sin  = abs(R[0, 1])
        new_w    = int(h * abs_sin + w * abs_cos)
        new_h    = int(h * abs_cos + w * abs_sin)
        R[0, 2] += (new_w * 0.5 - center[0])
        R[1, 2] += (new_h * 0.5 - center[1])
        center   = (new_w * 0.5, new_h * 0.5)  # W, H
    
    T = translate
    S = [math.radians(-shear[0]), math.radians(-shear[1])]
    M = np.float32([[R[0, 0]       , S[0] + R[0, 1], R[0, 2] + T[0] + (-S[0] * center[1])],
                    [S[1] + R[1, 0], R[1, 1]       , R[1, 2] + T[1] + (-S[1] * center[0])],
                    [0             , 0             , 1]])
    
    img = cv2.warpPerspective(img, M, (new_w, new_h))
    return img


@batch_image_processing
def affine(
    image        : TensorOrArray,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
) -> TensorOrArray:
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (TensorOrArray[C, H, W]):
            Image to be transformed.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale
        shear (FloatAnyT):
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
        image (TensorOrArray[C, H, W]):
            Transformed image.
    """
    if isinstance(image, Tensor):
        return _affine_tensor_image(
            image         = image,
            angle         = angle,
            translate     = translate,
            scale         = scale,
            shear         = shear,
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
			pad_mode      = pad_mode,
            fill          = fill,
        )
    elif isinstance(image, np.ndarray):
        return _affine_numpy_image(
            image         = image,
            angle         = angle,
            translate     = translate,
            scale         = scale,
            shear         = shear,
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
			pad_mode      = pad_mode,
            fill          = fill,
        )
    else:
        raise ValueError(f"Do not support {type(image)}.")


def affine_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Optional[ListOrTuple2T[int]] = None,
    interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
    keep_shape   : bool                         = True,
    pad_mode     : Union[PaddingMode, str]      = "constant",
    fill         : Optional[FloatAnyT]          = None,
    drop_ratio   : float                        = 0.0,
) -> tuple[TensorOrArray, TensorOrArray]:
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (TensorOrArray[C, H, W]):
            Image to be transformed.
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale
        shear (FloatAnyT):
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
        image (TensorOrArray[C, H, W]):
            Transformed image.
        box (TensorOrArray[B, 4]):
            Transformed box.
    """
    image_size = get_image_size(image)
    return \
        affine(
            image         = image,
            angle         = angle,
            translate     = translate,
            scale         = scale,
            shear         = shear,
            center        = center,
            interpolation = interpolation,
            keep_shape    = keep_shape,
            pad_mode      = pad_mode,
            fill          = fill,
        ), \
        affine_box(
            box        = box,
            image_size = image_size,
            angle      = angle,
            translate  = translate,
            scale      = scale,
            shear      = shear,
            center     = center,
            drop_ratio = drop_ratio,
        )


# MARK: - Modules

@TRANSFORMS.register(name="affine")
class Affine(nn.Module):
    """Apply affine transformation on the image keeping image center invariant.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Args:
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale
        shear (FloatAnyT):
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
        angle        : float,
        translate    : IntAnyT,
        scale        : float,
        shear        : FloatAnyT,
        center       : Optional[ListOrTuple2T[int]] = None,
        interpolation: InterpolationMode            = InterpolationMode.BILINEAR,
        keep_shape   : bool                         = True,
        pad_mode     : Union[PaddingMode, str]      = "constant",
        fill         : Optional[FloatAnyT]          = None,
    ):
        super().__init__()
        self.angle         = angle
        self.translate     = translate
        self.scale         = scale
        self.shear         = shear
        self.center        = center
        self.interpolation = interpolation
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
        return affine(
            image         = image,
            angle         = self.angle,
            translate     = self.translate,
            scale         = self.scale,
            shear         = self.shear,
            center        = self.center,
            interpolation = self.interpolation,
            fill          = self.fill,
        )
