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

import inspect
import math
import random
import sys
from copy import copy
from random import randint
from random import uniform
from typing import Any
from typing import Sequence
from typing import Union

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t
from torchvision.transforms.functional import _get_inverse_affine_matrix
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import resized_crop
from torchvision.transforms.functional import vflip

from one.core import batch_image_processing
from one.core import channel_last_processing
from one.core import Color
from one.core import error_console
from one.core import Float2T
from one.core import FloatAnyT
from one.core import get_image_center4
from one.core import get_image_hw
from one.core import get_image_size
from one.core import Int2Or3T
from one.core import Int2T
from one.core import Int3T
from one.core import IntAnyT
from one.core import InterpolationMode
from one.core import is_channel_last
from one.core import ListOrTuple2T
from one.core import ListOrTupleAnyT
from one.core import pad_image
from one.core import PaddingMode
from one.core import ScalarOrCollectionAnyT
from one.core import TensorOrArray
from one.core import to_channel_last
from one.core import to_size
from one.core import TRANSFORMS
from one.vision.filtering import adjust_gamma
from one.vision.shape import affine_box
from one.vision.shape import compute_single_box_iou
from one.vision.shape import hflip_box
from one.vision.shape import is_box_candidates
from one.vision.shape import scale_box
from one.vision.shape import vflip_box


# MARK: - Functional

def _affine_tensor_image(
    image        : Tensor,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
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
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
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
            If `True`, expands the output image to make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
            Default: `None`.

    Returns:
        image (Tensor[B, C, H, W]):
            Transformed image.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(
            f"`angle` must be `int` or `float`. But got: {type(angle)}."
        )
    if isinstance(angle, int):
        angle = float(angle)
    
    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if isinstance(translate, tuple):
        translate = list(translate)
    if not isinstance(translate, (list, tuple)):
        raise TypeError(
            f"`translate` must be `list` or `tuple`. "
            f"But got: {type(translate)}."
        )
    if not len(translate) == 2:
        raise ValueError(
            f"`translate` must be a sequence of length 2. "
            f"But got: {len(translate)}."
        )
    
    if isinstance(scale, int):
        scale = float(scale)
    if not scale >= 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")

    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(
            f"`shear` must be a single value or a sequence of length 2. "
            f"But got: {shear}."
        )
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if not len(shear) == 2:
        raise ValueError(
            f"`translate` must be a sequence of length 2. "
            f"But got: {len(shear)}."
        )
   
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
        pil_interpolation = InterpolationMode.pil_modes_mapping[interpolation]
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
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
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
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
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
            If `True`, expands the output image to make it large enough to
            hold the entire rotated image.
            If `False` or omitted, make the output image the same size as the
            input image.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation. Default: `True`.
        pad_mode (PaddingMode, str):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
            
    Returns:
        image (np.ndarray[C, H, W]):
            Transformed image.
    """
    if not image.ndim == 3:
        raise ValueError(f"`image.ndim` must be 3. But got: {image.ndim}.")
  
    if not isinstance(angle, (int, float)):
        raise TypeError(
            f"`angle` must be `int` or `float`. But got: {type(angle)}."
        )
    if isinstance(angle, int):
        angle = float(angle)
    
    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if not isinstance(translate, (list, tuple)):
        raise TypeError(
            f"`translate` must be `list` or `tuple`. "
            f"But got: {type(translate)}."
        )
    if isinstance(translate, tuple):
        translate = list(translate)
    if len(translate) != 2:
        raise ValueError(
            f"`translate` must be a sequence of length 2. "
            f"But got: {len(translate)}."
        )
    
    if isinstance(scale, int):
        scale = float(scale)
    if scale < 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")
   
    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(
            f"`shear` must be a single value or a sequence of length 2. "
            f"But got: {shear}."
        )
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(
            f"`translate` must be a sequence of length 2. "
            f"But got: {len(shear)}."
        )
    
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            f"`interpolation` must be a `InterpolationMode`. "
            f"But got: {type(interpolation)}."
        )
    
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


def _cast_squeeze_in(image: np.ndarray, req_dtypes: list[Any]) -> tuple[np.ndarray, bool, bool, Any]:
    need_expand = False
    # make image HWC
    if image.ndim == 4:
        image       = np.squeeze(image, axis=0)
        need_expand = True
    image = to_channel_last(image)

    out_dtype = image.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        image     = image.astype(req_dtype)
    return image, need_cast, need_expand, out_dtype


def _cast_squeeze_out(image: np.ndarray, need_cast: bool, need_expand: bool, out_dtype: Any) -> np.ndarray:
    if need_expand:
        image = np.expand_dims(image, axis=0)

    if need_cast:
        if out_dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
            # it is better to round before cast
            image = np.round(image)
        image = image.astype(out_dtype)

    return image


@batch_image_processing
def affine(
    image        : TensorOrArray,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0,
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


@batch_image_processing
def crop_zero_region(image: TensorOrArray) -> TensorOrArray:
    """Crop the zero region around the non-zero region in image.
    
    Args:
        image (TensorOrArray[C, H, W]):
            Image to with zeros background.
            
    Returns:
        image (TensorOrArray[C, H, W]):
            Cropped image.
    """
    if isinstance(image, Tensor):
        any   = torch.any
        where = torch.where
    elif isinstance(image, np.ndarray):
        any   = np.any
        where = np.where
    
    if is_channel_last(image):
        cols       = any(image, axis=0)
        rows       = any(image, axis=1)
        xmin, xmax = where(cols)[0][[0, -1]]
        ymin, ymax = where(rows)[0][[0, -1]]
        image      = image[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cols       = any(image, axis=1)
        rows       = any(image, axis=2)
        xmin, xmax = where(cols)[0][[0, -1]]
        ymin, ymax = where(rows)[0][[0, -1]]
        image      = image[:, ymin:ymax + 1, xmin:xmax + 1]
    return image


def horizontal_flip_image_box(
    image: TensorOrArray, box: TensorOrArray = ()
) -> tuple[TensorOrArray, TensorOrArray]:
    center = get_image_center4(image)
    if isinstance(image, Tensor):
        return F.hflip(image), hflip_box(box, center)
    elif isinstance(image, np.ndarray):
        return np.fliplr(image), hflip_box(box, center)
    else:
        raise ValueError(f"Do not support: {type(image)}")


def horizontal_shear(
    image        : TensorOrArray,
    magnitude    : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Shear image horizontally.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def horizontal_translate(
    image        : TensorOrArray,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Translate image in horizontal direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Horizontal translation (post-rotation translation)
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def horizontal_translate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def image_box_random_perspective(
    image      : np.ndarray,
    box        : np.ndarray = (),
    rotate     : float      = 10,
    translate  : float      = 0.1,
    scale      : float      = 0.1,
    shear      : float      = 10,
    perspective: float      = 0.0,
    border     : Int2T      = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    r"""Perform random perspective the image and the corresponding bounding box
    labels.

    Args:
        image (np.ndarray):
            Image of shape [H, W, C].
        box (np.ndarray):
            Bounding box labels where the box coordinates are located at:
            labels[:, 2:6]. Default: `()`.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (sequence):

    Returns:
        image_new (np.ndarray):
            Augmented image.
        box_new (np.ndarray):
            Augmented bounding boxes.
    """
    height    = image.shape[0] + border[0] * 2  # Shape of [H, W, C]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    box_new   = box.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T       = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image_new = cv2.warpPerspective(
                image_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image_new = cv2.warpAffine(
                image_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )

    # NOTE: Transform bboxes' coordinates
    n = len(box_new)
    if n:
        # NOTE: Warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = box_new[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)
        # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # Transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # Rescale
        else:  # Affine
            xy = xy[:, :2].reshape(n, 8)
        
        # NOTE: Create new boxes
        x  = xy[:, [0, 2, 4, 6]]
        y  = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        
        # NOTE: Filter candidates
        i = is_box_candidates(box_new[:, 2:6].T * s, xy.T)
        box_new = box_new[i]
        box_new[:, 2:6] = xy[i]
    
    return image_new, box_new


def letterbox_resize(
    image     : np.ndarray,
    size      : Union[Int2Or3T, None] = 768,
    stride    : int                   = 32,
    color     : Color                 = (114, 114, 114),
    auto      : bool                  = True,
    scale_fill: bool                  = False,
    scale_up  : bool                  = True
):
    """Resize image to a `stride`-pixel-multiple rectangle.
    
    For YOLOv5, stride = 32.
    For Scaled-YOLOv4, stride = 128
    
    References:
        https://github.com/ultralytics/yolov3/issues/232
        
    Args:
        image:
        size:
        stride:
        color:
        auto:
        scale_fill:
        scale_up:

    Returns:

    """
    # old_size = image.old_size[:2]  # current old_size [height, width]
    old_size = get_image_hw(image)
    
    if size is None:
        return image, None, None, None
    size = to_size(size)
    
    # Scale ratio (new / old)
    r = min(size[0] / old_size[0], size[1] / old_size[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio     = r, r  # width, height ratios
    new_unpad = int(round(old_size[1] * r)), int(round(old_size[0] * r))
    dw, dh    = size[1] - new_unpad[0], size[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh    = 0.0, 0.0
        new_unpad = (size[1], size[0])
        ratio     = size[1] / old_size[1], size[0] / old_size[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if old_size[::-1] != new_unpad:  # resize
        image = cv2.resize(src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return image, ratio, (dw, dh)


def lowhighres_images_random_crop(
    lowres : Tensor,
    highres: Tensor,
    size   : int,
    scale  : int
) -> tuple[Tensor, Tensor]:
    """Random cropping a pair of low and high resolution images."""
    lowres_left    = random.randint(0, lowres.shape[2] - size)
    lowres_right   = lowres_left   + size
    lowres_top     = random.randint(0, lowres.shape[1] - size)
    lowres_bottom  = lowres_top    + size
    highres_left   = lowres_left   * scale
    highres_right  = lowres_right  * scale
    highres_top    = lowres_top    * scale
    highres_bottom = lowres_bottom * scale
    lowres         = lowres[ :,  lowres_top:lowres_bottom,   lowres_left:lowres_right]
    highres        = highres[:, highres_top:highres_bottom, highres_left:highres_right]
    return lowres, highres


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
    
    
def paired_images_random_perspective(
    image1     : np.ndarray,
    image2     : np.ndarray = (),
    rotate     : float      = 10,
    translate  : float      = 0.1,
    scale      : float      = 0.1,
    shear      : float      = 10,
    perspective: float      = 0.0,
    border     : Sequence   = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask.

    Args:
        image1 (np.ndarray):
            Image.
        image2 (np.ndarray):
            Mask.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image1_new (np.ndarray):
            Augmented image.
        image2_new (np.ndarray):
            Augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height     = image1.shape[0] + border[0] * 2  # Shape of [HWC]
    width      = image1.shape[1] + border[1] * 2
    image1_new = image1.copy()
    image2_new = image2.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image1_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image1_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image1_new = cv2.warpPerspective(
                image1_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
            image2_new  = cv2.warpPerspective(
                image2_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image1_new = cv2.warpAffine(
                image1_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
            image2_new  = cv2.warpAffine(
                image2_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
    
    return image1_new, image2_new


def random_patch_numpy_image_box(
    canvas : TensorOrArray,
    patch  : ScalarOrCollectionAnyT[TensorOrArray],
    mask   : Union[ScalarOrCollectionAnyT[TensorOrArray], None] = None,
    id     : Union[ListOrTupleAnyT[int], None]                  = None,
    angle  : FloatAnyT = (0, 0),
    scale  : FloatAnyT = (1.0, 1.0),
    gamma  : FloatAnyT = (1.0, 1.0),
    overlap: float     = 0.1,
) -> tuple[TensorOrArray, TensorOrArray]:
    """Randomly place patches of small images over a large background image and
    generate accompany bounding boxes. Also, add some basic augmentation ops.
    
    References:
        https://datahacker.rs/012-blending-and-pasting-images-using-opencv/
    
    Args:
        canvas (TensorOrArray[C, H, W]):
            Background image to place patches over.
        patch (ScalarOrCollectionAnyT[TensorOrArray]):
            Collection of TensorOrArray[C, H, W] or a TensorOrArray[B, C, H, W]
            of small images.
        mask (ScalarOrCollectionAnyT[TensorOrArray]):
            Collection of TensorOrArray[C, H, W] or a TensorOrArray[B, C, H, W]
            of interested objects' masks in small images (black and white image).
        id (ListOrTupleAnyT[int], None):
            Bounding boxes' IDs.
        angle (FloatAnyT):
            Patches will be randomly rotated with angle in degree between
            `angle[0]` and `angle[1]`, clockwise direction. Default: `(0.0, 0.0)`.
        scale (FloatAnyT):
            Patches will be randomly scaled with factor between `scale[0]` and
            `scale[1]`. Default: `(1.0, 1.0)`.
        gamma (FloatAnyT):
            Gamma correction value used to augment the brightness of the objects
            between `gamma[0]` and `gamma[1]`. Default: `(1.0, 1.0)`.
        overlap (float):
            Overlapping ratio threshold.
            
    Returns:
        gen_image (TensorOrArray[C, H, W]):
            Generated image.
        box (TensorOrArray[N, 5], None):
            Bounding boxes of small patches. Boxes are expected to be in
            (id, x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """
    if type(patch) != type(mask):
        raise TypeError(
            f"`patch` and `mask` must have the same type. "
            f"But got: {type(patch)} != {type(mask)}."
        )
    if isinstance(patch, (Tensor, np.ndarray)) and isinstance(mask, (Tensor, np.ndarray)):
        if patch.shape != mask.shape:
            raise ValueError(
                f"`patch` and `mask` must have the same shape. "
                f"But got: {patch.shape} != {mask.shape}."
            )
        patch = list(patch)
        mask  = list(mask)
    if isinstance(patch, (list, tuple)) and isinstance(mask, (list, tuple)):
        if len(patch) != len(mask):
            raise ValueError(
                f"`patch` and `mask` must have the same length. "
                f"But got: {len(patch)} != {len(mask)}."
            )
    
    if isinstance(angle, (int, float)):
        angle = [-int(angle), int(angle)]
    if len(angle) == 1:
        angle = [-angle[0], angle[0]]
    
    if isinstance(scale, (int, float)):
        scale = [float(scale), float(scale)]
    if len(scale) == 1:
        scale = [scale, scale]

    if isinstance(gamma, (int, float)):
        gamma = [0.0, float(angle)]
    if len(gamma) == 1:
        gamma = [0.0, gamma]
    if not 0 < gamma[1] <= 1.0:
        raise ValueError(f"`gamma` must be between 0.0 and 1.0.")
    
    if mask is not None:
        # for i, (p, m) in enumerate(zip(patch, mask)):
        #     cv2.imwrite(f"{i}_image.png", p[:, :, ::-1])
        #     cv2.imwrite(f"{i}_mask.png",  m[:, :, ::-1])
        patch = [cv2.bitwise_and(p, m) for p, m in zip(patch, mask)]
        # for i, p in enumerate(patch):
        #     cv2.imwrite(f"{i}_patch.png", p[:, :, ::-1])

    if isinstance(id, (list, tuple)):
        if len(id) != len(patch):
            raise ValueError(
                f"`id` and `patch` must have the same length. "
                f"But got: {len(id)} != {len(patch)}."
            )
    
    canvas = copy(canvas)
    canvas = adjust_gamma(canvas, 2.0)
    h, w   = get_image_size(canvas)
    box    = np.zeros(shape=[len(patch), 5], dtype=np.float)
    for i, p in enumerate(patch):
        # Random scale
        s          = uniform(scale[0], scale[1])
        p_h0, p_w0 = get_image_size(p)
        p_h1, p_w1 = (int(p_h0 * s), int(p_w0 * s))
        p          = resize(image=p, size=(p_h1, p_w1))
        # Random rotate
        p          = rotate(p, angle=randint(angle[0], angle[1]), keep_shape=False)
        # p          = ndimage.rotate(p, randint(angle[0], angle[1]))
        p          = crop_zero_region(p)
        p_h, p_w   = get_image_size(p)
        # cv2.imwrite(f"{i}_rotate.png", p[:, :, ::-1])
        
        # Random place patch in canvas. Set ROI's x, y position.
        tries     = 0
        iou_thres = overlap
        while tries <= 10:
            x1  = randint(0, w - p_w)
            y1  = randint(0, h - p_h)
            x2  = x1 + p_w
            y2  = y1 + p_h
            roi = canvas[y1:y2, x1:x2]
            
            if id is not None:
                b = np.array([id[i], x1, y1, x2, y2], dtype=np.float)
            else:
                b = np.array([-1, x1, y1, x2, y2], dtype=np.float)
                
            max_iou = max([compute_single_box_iou(b[1:5], j[1:5]) for j in box])
            if max_iou <= iou_thres:
                box[i] = b
                break
            
            tries += 1
            if tries == 10:
                iou_thres += 0.1
        
        # Blend patch into canvas
        p_blur    = cv2.medianBlur(p, 3)  # Blur to remove noise around the edges of objects
        p_gray    = cv2.cvtColor(p_blur, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(p_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv  = cv2.bitwise_not(mask)
        bg        = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg        = cv2.bitwise_and(p,   p,   mask=mask)
        dst       = cv2.add(bg, fg)
        roi[:]    = dst
        # cv2.imwrite(f"{i}_gray.png", p_gray)
        # cv2.imwrite(f"{i}_threshold.png", mask)
        # cv2.imwrite(f"{i}_maskinv.png", mask_inv)
        # cv2.imwrite(f"{i}_bg.png", bg[:, :, ::-1])
        # cv2.imwrite(f"{i}_fg.png", fg[:, :, ::-1])
        # cv2.imwrite(f"{i}_dst.png", dst[:, :, ::-1])
    
    # Adjust brightness via Gamma correction
    g      = uniform(gamma[0], gamma[1])
    canvas = adjust_gamma(canvas, g)
    return canvas, box


def random_perspective(
    image      : np.ndarray,
    rotate     : float    = 10,
    translate  : float    = 0.1,
    scale      : float    = 0.1,
    shear      : float    = 10,
    perspective: float    = 0.0,
    border     : Sequence = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask labels.

    Args:
        image (np.ndarray):
            Image.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image_new (np.ndarray):
            Augmented image.
        mask_labels_new (np.ndarray):
            Augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T       = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image_new = cv2.warpPerspective(
                image_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image_new = cv2.warpAffine(
                image_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
    
    return image_new


def resize(
    image        : Union[Tensor, np.ndarray, PIL.Image],
    size         : Union[Int2Or3T, None] = None,
    interpolation: InterpolationMode     = InterpolationMode.LINEAR,
    max_size     : Union[int, None]      = None,
    antialias    : Union[bool, None]     = None
) -> Union[Tensor, np.ndarray, PIL.Image]:
    """Resize an image. Adapted from:
    `torchvision.transforms.functional.resize()`
    
    Args:
        image (Tensor, np.ndarray, PIL.Image.Image):
            Image of shape [H, W, C].
        size (Int2Or3T[H, W, C*], None):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, None):
        
        antialias (bool, None):

    Returns:
        resize (Tensor, np.ndarray, PIL.Image.Image):
            Resized image of shape [H, W, C].
    """
    if size is None:
        return image
    if isinstance(image, Tensor):
        if interpolation is InterpolationMode.LINEAR:
            interpolation = InterpolationMode.BILINEAR
        return resize_tensor_image(image, size, interpolation, max_size, antialias)
    elif isinstance(image, np.ndarray):
        if interpolation is InterpolationMode.BILINEAR:
            interpolation = InterpolationMode.LINEAR
        return resize_numpy_image(image, size, interpolation, max_size, antialias)
    else:
        return resize_pil_image(image, size, interpolation, max_size, antialias)


@batch_image_processing
def resize_numpy_image(
    image        : np.ndarray,
    size         : Union[Int2Or3T, None] = None,
    interpolation: InterpolationMode     = InterpolationMode.LINEAR,
    max_size     : Union[int, None]      = None,
    antialias    : Union[bool, None]     = None
) -> np.ndarray:
    """Resize a numpy image. Adapted from:
    `torchvision.transforms.functional_tensor.resize()`
    
    Args:
        image (np.ndarray[C, H, W]):
            Image to be resized.
        size (Int2Or3T[H, W, C*], None):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, None):
        
        antialias (bool, None):

    Returns:
        resize (np.ndarray[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")
    cv_interpolation = InterpolationMode.cv_modes_mapping[interpolation]
    if cv_interpolation not in list(InterpolationMode.cv_modes_mapping.values()):
        raise ValueError(
            "This interpolation mode is unsupported with np.ndarray input"
        )

    if size is None:
        return image
    size = to_size(size)[::-1]  # W, H
    
    if antialias is None:
        antialias = False
    if antialias and cv_interpolation not in [cv2.INTER_LINEAR, cv2.INTER_CUBIC]:
        raise ValueError(
            "Antialias option is supported for linear and cubic "
            "interpolation modes only"
        )

    w, h = get_image_hw(image)
    # Specified size only for the smallest edge
    if isinstance(size, int) or len(size) == 1:
        short, long         = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        if short == requested_new_short:
            return image

        new_short = requested_new_short,
        new_long  = int(requested_new_short * long / short)
        
        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the "
                    f"requested size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short = int(max_size * new_short / new_long),
                new_long  = max_size
        
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        
    else:  # specified both h and w
        new_w, new_h = size[0], size[1]
    
    image, need_cast, need_expand, out_dtype = _cast_squeeze_in(
        image, [np.float32, np.float64]
    )
    
    image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv_interpolation)
    
    if cv_interpolation == cv2.INTER_CUBIC and out_dtype == np.uint8:
        image = np.clip(image, 0, 255)
    
    image = _cast_squeeze_out(
        image, need_cast=need_cast, need_expand=need_expand, out_dtype=out_dtype
    )
    
    return image


def resize_pil_image(
    image        : PIL.Image.Image,
    size         : Union[Int2Or3T, None] = None,
    interpolation: InterpolationMode     = InterpolationMode.BILINEAR,
    max_size     : Union[int, None]      = None,
    antialias    : Union[bool, None]     = None
) -> PIL.Image:
    """Resize a pil image. Adapted from:
    `torchvision.transforms.functional_pil.resize()`
    
    Args:
        image (PIL.Image.Image[H, W, C]):
            Image.
        size (Int2Or3T[H, W, C*], None):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, None):
        
        antialias (bool, None):

    Returns:
        resize (PIL.Image.Image[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")
    
    if antialias is not None and not antialias:
        error_console.log(
            "Anti-alias option is always applied for PIL Image input. "
            "Argument antialias is ignored."
        )
    pil_interpolation = InterpolationMode.pil_modes_mapping()[interpolation]

    if size is None:
        return image
    size = to_size(size)  # H, W
    
    return F_pil.resize(
        image         = image,
        size          = size[::-1],  # W, H
        interpolation = pil_interpolation,
        max_size      = max_size,
        antialias     = antialias
    )


def resize_tensor_image(
    image        : Tensor,
    size         : Union[Int2Or3T, None] = None,
    interpolation: InterpolationMode     = InterpolationMode.BILINEAR,
    max_size     : Union[int, None]      = None,
    antialias    : Union[bool, None]     = None
) -> Tensor:
    """Resize a tensor image. Adapted from:
    `torchvision.transforms.functional_tensor.resize()`
    
    Args:
        image (Tensor[H, W, C]):
            Image.
        size (Int2Or3T[H, W, C*], None):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, None):
        
        antialias (bool, None):

    Returns:
        resize (Tensor[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if size is None:
        return image
    size = to_size(size)  # H, W

    return F_t.resize(
        img           = image,
        size          = size,  # H, W
        interpolation = interpolation.value,
        max_size      = max_size,
        antialias     = antialias
    )


def rotate(
    image        : TensorOrArray,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> Tensor:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].
    
    Args:
        image (TensorOrArray[*, C, H, W]):
            Image to be rotated.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def rotate_horizontal_flip(
    image        : TensorOrArray,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Rotate a tensor image or a batch of tensor images and then horizontally
    flip.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def rotate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def rotate_vertical_flip(
    image        : TensorOrArray,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Rotate a tensor image or a batch of tensor images and then vertically
    flip.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def scale(
    image        : TensorOrArray,
    factor       : Float2T,
    interpolation: InterpolationMode       = InterpolationMode.BILINEAR,
    antialias    : bool                    = False,
    keep_shape   : bool                    = False,
    pad_mode     : Union[PaddingMode, str] = "constant",
    fill         : Union[FloatAnyT, None]  = None,
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
        antialias (bool, None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (TensorOrArray[B, C, H * factor, W * factor]):
            Rescaled image with the shape as the specified size.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if pad_mode not in ("constant", PaddingMode.CONSTANT):
        raise ValueError(
            f"Current only support pad_mode == 'constant'. "
            f"But got: {pad_mode}."
        )
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
    fill         : Union[FloatAnyT, None]  = None,
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
        antialias (bool, None):
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
        fill (FloatAnyT, None):
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
        raise ValueError(
            f"Current only support pad_mode == 'constant'."
            f"But got: {pad_mode}."
        )
    
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


def shear(
    image        : TensorOrArray,
    magnitude    : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def translate(
    image        : TensorOrArray,
    magnitude    : Int2T,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Translate image in vertical and horizontal direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (Int2T):
            Horizontal and vertical translations (post-rotation translation).
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def vertical_flip_image_box(
    image: TensorOrArray, box: TensorOrArray = ()
) -> tuple[TensorOrArray, TensorOrArray]:
    center = get_image_center4(image)
    if isinstance(image, Tensor):
        return F.vflip(image), vflip_box(box, center)
    elif isinstance(image, np.ndarray):
        return np.flipud(image), vflip_box(box, center)
    else:
        raise ValueError(f"Do not support: {type(image)}")


def vertical_shear(
    image        : TensorOrArray,
    magnitude    : float,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Shear image vertically.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def vertical_translate(
    image        : TensorOrArray,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
) -> TensorOrArray:
    """Translate image in vertical direction.
    
    Args:
        image (TensorOrArray[B, C, H, W]):
            Image to transform.
        magnitude (int):
            Vertical translation (post-rotation translation)
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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


def vertical_translate_image_box(
    image        : TensorOrArray,
    box          : TensorOrArray,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None] = None,
    interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
    keep_shape   : bool                            = True,
    pad_mode     : Union[PaddingMode, str]         = "constant",
    fill         : Union[FloatAnyT, None]          = None,
    drop_ratio   : float                           = 0.0
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
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
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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


@TRANSFORMS.register(name="center_crop")
class CenterCrop(nn.Module):
    """Crops the given image at the center.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded
    with 0 and then cropped.

    Args:
        output_size (Int2T):
            [height, width] of the crop box. If int or sequence with single int,
            it is used for both directions.
    """
    
    def __init__(self, output_size: Int2T):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return center_crop(image, self.output_size)
    
    
@TRANSFORMS.register(name="crop")
class Crop(nn.Module):
    """Crop the given image at specified location and output size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded
    with 0 and then cropped.

    Args:
        top (int):
            Vertical component of the top left corner of the crop box.
        left (int):
            Horizontal component of the top left corner of the crop box.
        height (int):
            Height of the crop box.
        width (int):
            Width of the crop box.
    """
    
    def __init__(self, top: int, left: int, height: int, width: int):
        super().__init__()
        self.top    = top
        self.left   = left
        self.height = height
        self.width  = width
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return crop(image, self.top, self.left, self.height, self.width)


@TRANSFORMS.register(name="hflip")
@TRANSFORMS.register(name="horizontal_flip")
class HorizontalFlip(nn.Module):
    """Horizontally flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

    Examples:
        >>> hflip = Hflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> hflip(input)
        image([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [1., 1., 0.]]]])
    """
    
    # MARK: Magic Functions
    
    def __repr__(self):
        return self.__class__.__name__
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return hflip(image)


@TRANSFORMS.register(name="hflip_image_box")
@TRANSFORMS.register(name="horizontal_flip_image_box")
class HorizontalFlipImageBox(nn.Module):

    # MARK: Magic Functions
    
    def __repr__(self):
        return self.__class__.__name__
    
    # MARK: Forward Pass
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        return horizontal_flip_image_box(image=image, box=box)
    

@TRANSFORMS.register(name="hshear")
@TRANSFORMS.register(name="horizontal_shear")
class HorizontalShear(nn.Module):
    """
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : float,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return horizontal_shear(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill
        )


@TRANSFORMS.register(name="htranslate")
@TRANSFORMS.register(name="horizontal_translate")
class HorizontalTranslate(nn.Module):
    """
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return horizontal_translate(
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
class HorizontalTranslateImageBox(nn.Module):
    """
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return horizontal_translate_image_box(
            image         = image,
            box           = box,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="lowhighres_images_random_crop")
class LowHighResImagesRandomCrop(nn.Module):
    """Random cropping a pair of low and high resolution images.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Args:
        size (int):
            The patch size.
        scale (int):
            Scale factor.
    """

    def __init__(self, size: int, scale: int):
        super().__init__()
        self.size  = size
        self.scale = scale
    
    def forward(self, low_res: Tensor, high_res: Tensor) -> tuple[Tensor, Tensor]:
        """
        
        Args:
            low_res (PIL Image or Tensor):
                Low resolution image.
            high_res (PIL Image or Tensor):
                High resolution image.
            
        Returns:
            Cropped images.
        """
        return lowhighres_images_random_crop(
            lowres=low_res, highres=high_res, size=self.size, scale=self.scale
        )


@TRANSFORMS.register(name="padded_scale")
class PaddedScale(nn.Module):
    
    def __init__(self, ratio: float = 1.0, same_shape: bool = False):
        super().__init__()
        self.ratio      = ratio
        self.same_shape = same_shape
    
    def forward(self, image: Tensor) -> Tensor:
        return padded_scale(image, self.ratio, self.same_shape)


@TRANSFORMS.register(name="resize")
class Resize(nn.Module):
    r"""Resize the input image to the given size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    .. warning::
        Output image might be different depending on its type: when
        downsampling, the interpolation of PIL images and tensors is slightly
        different, because PIL applies antialiasing. This may lead to
        significant differences in the performance of a network.
        Therefore, it is preferable to train and serve a model with the same
        input types. See also below the `antialias` parameter, which can help
        making the output of PIL images and tensors closer.

    Args:
        size (Int3T):
            Desired output size. If size is a sequence like [H, W], the output
            size will be matched to this. If size is an int, the smaller edge
            of the image will be matched to this number maintaining the aspect
            ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{
            width}}, \text{size}\right)`.
            .. note::
                In torchscript mode size as single int is not supported, use a
                sequence of length 1: `[size, ]`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        max_size (int, None):
            Maximum allowed for the longer edge of the resized image: if
            the longer edge of the image is greater than `max_size` after being
            resized according to `size`, then the image is resized again so
            that the longer edge is equal to `max_size`. As a result, `size`
            might be overruled, i.e the smaller edge may be shorter than `size`.
            This is only supported if `size` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, None):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.

            .. warning::
                There is no autodiff support for `antialias=True` option with
                input `img` as Tensor.
    """
    
    def __init__(
        self,
        size         : Union[Int3T, None],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size     : Union[int, None]  = None,
        antialias    : Union[bool, None] = None
    ):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
    
    def forward(
        self, image: Union[Tensor, np.ndarray, PIL.Image.Image]
    ) -> Union[Tensor, np.ndarray, PIL.Image.Image]:
        """

        Args:
            image (Tensor, np.ndarray, PIL.Image.Image):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.

        Returns:
            image (Tensor, np.ndarray, PIL.Image.Image):
                Resized image.
        """
        return resize(
            image, self.size, self.interpolation, self.max_size, self.antialias,
        )


@TRANSFORMS.register(name="resized_crop")
class ResizedCrop(nn.Module):
    """Crop the given image and resize it to desired size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        top (int):
            Vertical component of the top left corner of the crop box.
        left (int):
            Horizontal component of the top left corner of the crop box.
        height (int):
            Height of the crop box.
        width (int):
            Width of the crop box.
        size (list[int]):
            Desired output size. Same semantics as `resize`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
    """
    
    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return resized_crop(
            image, self.top, self.left, self.height, self.width, self.size,
            self.interpolation
        )


@TRANSFORMS.register(name="rotate")
class Rotate(nn.Module):
    """Rotate a tensor image or a batch of tensor images.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
class RotateHorizontalFlip(nn.Module):
    """Rotate a tensor image or a batch of tensor images and then horizontally
    flip.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return rotate_horizontal_flip(
            image         = image,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="rotate_vflip")
class RotateVerticalFlip(nn.Module):
    """Rotate a tensor image or a batch of tensor images and then vertically
    flip.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return rotate_vertical_flip(
            image         = image,
            angle         = self.angle,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


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
        antialias (bool, None):
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
        fill (FloatAnyT, None):
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
        fill         : Union[FloatAnyT, None]  = None,
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
        antialias (bool, None):
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
        fill (FloatAnyT, None):
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
        fill         : Union[FloatAnyT, None]  = None,
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


@TRANSFORMS.register(name="shear")
class Shear(nn.Module):
    """
    
    Args:
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x axis, while the second value
            corresponds to a shear parallel to the y axis.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : list[float],
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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


@TRANSFORMS.register(name="translate")
class Translate(nn.Module):
    """
    
    Args:
        magnitude (Int2T):
            Horizontal and vertical translation magnitude.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : Int2T,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : Int2T,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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


@TRANSFORMS.register(name="vflip")
@TRANSFORMS.register(name="vertical_flip")
class VerticalFlip(nn.Module):
    """Vertically flip a tensor image or a batch of tensor images. Input must
    be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

    Examples:
        >>> vflip = Vflip()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> vflip(input)
        image([[[[0., 1., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """
    
    # MARK: Magic Functions
    
    def __repr__(self):
        return self.__class__.__name__
    
    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        return vflip(image)
    
    
@TRANSFORMS.register(name="vflip_image_box")
@TRANSFORMS.register(name="vertical_flip_image_box")
class VerticalFlipImageBox(nn.Module):
    
    # MARK: Magic Functions
    
    def __repr__(self):
        return self.__class__.__name__
    
    # MARK: Forward Pass
    
    def forward(
        self, image: TensorOrArray, box: TensorOrArray
    ) -> tuple[TensorOrArray, TensorOrArray]:
        return vertical_flip_image_box(image=image, box=box)


@TRANSFORMS.register(name="yshear")
@TRANSFORMS.register(name="vertical_shear")
class VerticalShear(nn.Module):
    """
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : float,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return vertical_shear(
            image         = image,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill
        )


@TRANSFORMS.register(name="vtranslate")
@TRANSFORMS.register(name="vertical_translate")
class VerticalTranslate(nn.Module):
    """
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return vertical_translate(
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
class VerticalTranslateImageBox(nn.Module):
    """
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
        center (ListOrTuple2T[int], None):
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
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None] = None,
        interpolation: InterpolationMode               = InterpolationMode.BILINEAR,
        keep_shape   : bool                            = True,
        pad_mode     : Union[PaddingMode, str]         = "constant",
        fill         : Union[FloatAnyT, None]          = None,
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
        return vertical_translate_image_box(
            image         = image,
            box           = box,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
