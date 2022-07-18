#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import sys
from copy import copy
from copy import deepcopy
from typing import Union

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F
from multipledispatch import dispatch
from PIL import ExifTags
from torch import nn
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t

from one.core import batch_image_processing
from one.core import Color
from one.core import FloatAnyT
from one.core import Int2Or3T
from one.core import Int2T
from one.core import Int3T
from one.core import IntAnyT
from one.core import InterpolationMode
from one.core import ListOrTuple2T
from one.core import PaddingMode
from one.core import TensorOrArray
from one.core import to_size
from one.core import TRANSFORMS
from one.core.rich import error_console
from one.math import make_divisible
from one.vision.shape import affine_box
from one.vision.shape import hflip_box
from one.vision.shape import vflip_box
from one.vision.transformation.utils import Transform

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


# MARK: - Functional

def _affine_tensor_image(
    image        : Tensor,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.

    Args:
        image (Tensor[..., C, H, W]):
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
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
            Default: `None`.

    Returns:
        image (Tensor[..., C, H, W]):
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
    
    if not isinstance(interpolation, InterpolationMode):
        interpolation = InterpolationMode.from_value(interpolation)
    
    img    = image.clone()
    h, w   = get_image_size(img)
    center = (h * 0.5, w * 0.5) if center is None else center  # H, W
    center = tuple(center[::-1])  # W, H
    
    if not isinstance(image, Tensor):
        # center = (img_size[0] * 0.5 + 0.5, img_size[1] * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        matrix = F._get_inverse_affine_matrix(
            center    = center,
            angle     = angle,
            translate = translate,
            scale     = scale,
            shear     = shear
        )
        pil_interpolation = InterpolationMode.pil_modes_mapping[interpolation]
        return F_pil.affine(
            image         = image,
            matrix        = matrix,
            interpolation = pil_interpolation,
            fill          = fill
        )

    # If keep shape, find the new width and height bounds
    if not keep_shape:
        matrix  = F._get_inverse_affine_matrix(
            center    = [0, 0],
            angle     = angle,
            translate = [0, 0],
            scale     = 1.0,
            shear     = [0.0, 0.0]
        )
        abs_cos = abs(matrix[0])
        abs_sin = abs(matrix[1])
        new_w   = int(h * abs_sin + w * abs_cos)
        new_h   = int(h * abs_cos + w * abs_sin)
        image   = pad_image(image, pad_size=(new_h, new_w))
    
    translate_f = [1.0 * t for t in translate]
    matrix      = F._get_inverse_affine_matrix(
        center    = [0, 0],
        angle     = angle,
        translate = translate_f,
        scale     = scale,
        shear     = shear
    )
    return F_t.affine(
        img           = image,
        matrix        = matrix,
        interpolation = interpolation.value,
        fill          = fill
    )


def add_weighted(
    image1: Tensor,
    alpha : float,
    image2: Tensor,
    beta  : float,
    gamma : float = 0.0,
) -> Tensor:
    """Calculate the weighted sum of two Tensors.
    
    Function calculates the weighted sum of two Tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1 (Tensor[..., C, H, W]):
            First image Tensor.
        alpha (float):
            Weight of the image1 elements.
        image2 (Tensor[..., C, H, W]):
            Second image Tensor of same shape as `src1`.
        beta (float):
            Weight of the image2 elements.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.

    Returns:
        output (Tensor[..., C, H, W]):
            Weighted Tensor.
    """
    if not isinstance(image1, Tensor):
        raise TypeError(f"`image1` must be a `Tensor`. But got: {type(image1)}.")
    if not isinstance(image2, Tensor):
        raise TypeError(f"`image2` must be a `Tensor`. But got: {type(image2)}.")
    if image1.shape != image2.shape:
        raise ValueError(
            f"`image1` and `image2` must have the same shape. "
            f"But got: {image1.shape} != {image2.shape}."
        )
    if not isinstance(alpha, float):
        raise TypeError(f"`alpha` must be a `float`. But got: {type(alpha)}.")
    if not isinstance(beta, float):
        raise TypeError(f"`beta` must be a `float`. But got: {type(beta)}.")
    if not isinstance(gamma, float):
        raise TypeError(f"`gamma` must be a `float`. But got: {type(gamma)}.")
    
    bound = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    output = output.clamp(0, bound).to(image1.dtype)
    return output


def affine(
    image        : Union[Tensor, PIL.Image],
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (Tensor[..., C, H, W]):
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
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
            Transformed image.
    """
    if isinstance(image, (Tensor, PIL.Image)):
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
    else:
        raise ValueError(
            f"`image` must be a `Tensor` or `PIL.Image`. "
            f"But got: {type(image)}."
        )


def affine_image_box(
    image        : Tensor,
    box          : Tensor,
    angle        : float,
    translate    : IntAnyT,
    scale        : float,
    shear        : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be transformed.
        box (Tensor[N, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
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
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[C, H, W]):
            Transformed image.
        box (Tensor[N, 4]):
            Transformed box.
    """
    if box.ndim != 2:
        raise ValueError(
            f"R`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
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


def blend(
    image1: Tensor,
    image2: Tensor,
    alpha : float,
    gamma : float = 0.0
) -> Tensor:
    """Blends 2 images together.
    
    output = image1 * alpha + image2 * beta + gamma

    Args:
        image1 (Tensor[..., C, H, W]):
            Source image.
        image2 (Tensor[..., C, H, W]):
            Image we want to overlay on top of `image1`.
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.

    Returns:
        blend (Tensor[..., C, H, W]):
            Blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma
    )


def check_image_size(size: Int2Or3T, stride: int = 32) -> int:
    """Verify image size is a multiple of stride and return the new size.
    
    Args:
        size (Int2Or3T):
            Image size of shape [C*, H, W].
        stride (int):
            Stride. Default: `32`.
    
    Returns:
        new_size (int):
            Appropriate size.
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:    # [C, H, W]
            size = size[1]
        elif len(size) == 2:  # [H, W]
            size = size[0]
        
    new_size = make_divisible(size, int(stride))  # ceil gs-multiple
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


@batch_image_processing
def crop_zero_region(image: Tensor) -> Tensor:
    """Crop the zero region around the non-zero region in image.
    
    Args:
        image (Tensor[C, H, W]):
            Image to with zeros background.
            
    Returns:
        image (Tensor[C, H, W]):
            Cropped image.
    """
    if not isinstance(image, Tensor):
        raise ValueError(f"`image` must be a `Tensor`. But got: {type(image)}.")
    if is_channel_last(image):
        cols       = torch.any(image, dim=0)
        rows       = torch.any(image, dim=1)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cols       = torch.any(image, dim=1)
        rows       = torch.any(image, dim=2)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[:, ymin:ymax + 1, xmin:xmax + 1]
    return image


def denormalize(
    image: Tensor,
    mean : Union[Tensor, float],
    std  : Union[Tensor, float]
) -> Tensor:
    """Denormalize an image Tensor with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
        mean (Tensor[..., C, H, W], float):
            Mean for each channel.
        std (Tensor[..., C, H, W], float):
            Standard deviations for each channel.

    Returns:
        output (Tensor[..., C, H, W]):
            Denormalized image with same size as input.

    Examples:
        >>> x   = torch.rand(1, 4, 3, 3)
        >>> output = denormalize(x, 0.0, 255.)
        >>> output.shape
        torch.Size([1, 4, 3, 3])

        >>> x    = torch.rand(1, 4, 3, 3, 3)
        >>> mean = torch.zeros(1, 4)
        >>> std  = 255. * torch.ones(1, 4)
        >>> output  = denormalize(x, mean, std)
        >>> output.shape
        torch.Size([1, 4, 3, 3, 3])
    """
    shape = image.shape

    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=image.device, dtype=image.dtype)
    if isinstance(std, float):
        std  = torch.tensor([std] * shape[1], device=image.device, dtype=image.dtype)
    if not isinstance(image, Tensor):
        raise TypeError(f"`data` should be a `Tensor`. But got: {type(image)}")
    if not isinstance(mean, Tensor):
        raise TypeError(f"`mean` should be a `Tensor`. But got: {type(mean)}")
    if not isinstance(std, Tensor):
        raise TypeError(f"`std` should be a `Tensor`. But got: {type(std)}")

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != image.shape[-3] and mean.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"`mean` and `data` must have the same shape. "
                f"But got: {mean.shape} and {image.shape}."
            )

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != image.shape[-3] and std.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"`std` and `data` must have the same shape. "
                f"But got: {std.shape} and {image.shape}."
            )

    mean = torch.as_tensor(mean, device=image.device, dtype=image.dtype)
    std  = torch.as_tensor(std,  device=image.device, dtype=image.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std  = std[...,  :, None]

    output = (image.view(shape[0], shape[1], -1) * std) + mean
    return output.view(shape)


@dispatch(Tensor)
def denormalize_naive(image: Tensor) -> Tensor:
    """Naively denormalize an image Tensor.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
    
    Returns:
        output (Tensor[..., C, H, W]):
            Normalized image Tensor.
    """
    return torch.clamp(image * 255, 0, 255).to(torch.uint8)
    

@dispatch(list)
def denormalize_naive(image: list[Tensor]) -> list:
    """Naively denormalize a list of image Tensor.
    
    Args:
        image (list[Tensor[..., C, H, W]]):
            List of image Tensor.
    
    Returns:
        output (list[Tensor[..., C, H, W]]):
            Normalized list of image Tensors.
    """
    if all(i.ndim == 3 for i in image):
        return list(denormalize_naive(torch.stack(image)))
    elif all(i.ndim == 4 for i in image):
        return [denormalize_naive(i) for i in image]
    else:
        raise TypeError(f"`image` must be a list of `Tensor`.")


@dispatch(tuple)
def denormalize_naive(image: tuple) -> tuple:
    """Naively denormalize a tuple of image Tensor.
    
    Args:
        image (tuple[Tensor[..., C, H, W]]):
            Tuple of image Tensor.
    
    Returns:
        output (tuple[Tensor[..., C, H, W]]):
            Normalized tuple of image Tensors.
    """
    return tuple(denormalize_naive(list(image)))


@dispatch(dict)
def denormalize_naive(image: dict) -> dict:
    """Naively denormalize a dictionary of image Tensor.
    
    Args:
        image (dict):
            Dictionary of image Tensor.
    
    Returns:
        output (dict):
            Normalized dictionary of image Tensors.
    """
    if not all(isinstance(v, (Tensor, list, tuple)) for k, v in image.items()):
        raise TypeError(
            f"`image` must be a `dict` of `Tensor`, `list`, or `tuple`."
        )
    for k, v in image.items():
        image[k] = denormalize_naive(v)
    return image


def get_exif_size(image: PIL.Image) -> Int2T:
    """Return the exif-corrected PIL size.
    
    Args:
        image (PIL.Image):
            Image.
            
    Returns:
        size (Int2T[H, W]):
            Image size.
    """
    size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = (size[1], size[0])
        elif rotation == 8:  # rotation 90
            size = (size[1], size[0])
    except:
        pass
    return size[1], size[0]


def get_image_center(image: Tensor) -> Tensor:
    """Get image center as  (x=h/2, y=w/2).
    
    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
   
    Returns:
        center (Tensor[2]):
            Image center as (x=h/2, y=w/2).
    """
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2])


def get_image_center4(image: Tensor) -> Tensor:
    """Get image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    
    Args:
        image (Tensor[..., C, H, W]):
            Image.
   
    Returns:
        center (Tensor[4]):
            Image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    """
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    

def get_image_hw(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int2T:
    """Returns the size of an image as [H, W].
    
    Args:
        image (Tensor, np.ndarray, PIL Image):
            Image.
   
    Returns:
        size (Int2T):
            Image size as [H, W].
    """
    if isinstance(image, (Tensor, np.ndarray)):
        if is_channel_first(image):  # [.., C, H, W]
            return [image.shape[-2], image.shape[-1]]
        else:  # [.., H, W, C]
            return [image.shape[-3], image.shape[-2]]
    elif F._is_pil_image(image):
        return list(image.size)
    else:
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image. "
            f"But got: {type(image)}."
        )
    
    
get_image_size = get_image_hw


def get_image_shape(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int3T:
    """Returns the shape of an image as [H, W, C].

    Args:
        image (Tensor, np.ndarray, PIL Image):
            Image.

    Returns:
        shape (Int3T):
            Image shape as [C, H, W].
    """
    if isinstance(image, (Tensor, np.ndarray)):
        if is_channel_first(image):  # [.., C, H, W]
            return [image.shape[-3], image.shape[-2], image.shape[-1]]
        else:  # [.., H, W, C]
            return [image.shape[-1], image.shape[-3], image.shape[-2]]
    elif F._is_pil_image(image):
        return list(image.size)
    else:
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image`. "
            f"But got: {type(image)}."
        )


def get_num_channels(image: TensorOrArray) -> int:
    """Get number of channels of the image.
    
    Args:
        image (Tensor, np.ndarray):
            Image.

    Returns:
        num_channels (int):
            Image channels.
    """
    if not isinstance(image, (Tensor, np.ndarray)):
        raise TypeError(
            f"`image` must be a `Tensor` or `np.ndarray`. "
            f"But got: {type(image)}."
        )
    if image.ndim == 4:
        if is_channel_first(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
        return c
    elif image.ndim == 3:
        if is_channel_first(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
        return c
    else:
        raise ValueError(
            f"`image.ndim` must be == 3 or 4. But got: {image.ndim}."
        )


def horizontal_flip_image_box(
    image: Tensor, box: Tensor = ()
) -> tuple[Tensor, Tensor]:
    """Horizontally flip images and bounding boxes.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image.
        box (Tensor[N, 4):
            Box.
    
    Returns:
        image (Tensor[..., C, H, W]):
            Flipped image.
        box (Tensor[N, 4):
            Flipped box.
    """
    if box.ndim != 2:
        raise ValueError(
            f"`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
    center = get_image_center4(image)
    return F.hflip(image), hflip_box(box, center)


def horizontal_shear(
    image        : Tensor,
    magnitude    : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Shear image horizontally.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.NEAREST`.
            If input is Tensor, only `InterpolationMode.BILINEAR`,
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    
    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Translate image in horizontal direction.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (int):
            Horizontal translation (post-rotation translation)
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    box          : Tensor,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0
) -> tuple[Tensor, Tensor]:
    """Translate images and bounding boxes in horizontal direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor[..., C, H, W]):
            Image to be translated.
        box (Tensor[N, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (int):
            Horizontal translation (post-rotation translation).
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Translated image with the shape as the specified size.
        box (Tensor[N, 4]):
            Translated boxes.
    """
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


def is_channel_first(image: TensorOrArray) -> bool:
    """Return `True` if the image is in channel first format."""
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    raise ValueError(
        f"`image.ndim` must be == 3, 4, or 5. But got: {image.ndim}."
    )


def is_channel_last(image: TensorOrArray) -> bool:
    """Return `True` if the image is in channel last format."""
    return not is_channel_first(image)


def is_integer_image(image: TensorOrArray) -> bool:
    """Return `True` if the given image is integer-encoded."""
    c = get_num_channels(image)
    if c == 1:
        return True
    return False


def is_normalized(image: TensorOrArray) -> TensorOrArray:
    """Return `True` if the given image is normalized."""
    if isinstance(image, Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f"`image` must be a `Tensor` or `np.ndarray`. "
            f"But got: {type(image)}."
        )


def is_one_hot_image(image: TensorOrArray) -> bool:
    """Return `True` if the given image is one-hot encoded."""
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


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
    
    Notes:
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
        image = cv2.resize(
            src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR
        )
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image       = cv2.copyMakeBorder(
        src        = image,
        top        = top,
        bottom     = bottom,
        left       = left,
        right      = right,
        borderType = cv2.BORDER_CONSTANT,
        value      = color
    )  # add border
    
    return image, ratio, (dw, dh)


def normalize_min_max(
    image  : Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0,
    eps    : float = 1e-6
) -> Tensor:
    """Normalise an image/video image by MinMax and re-scales the value
    between a range.

    Args:
        image (Tensor[..., C, H, W]):
            Image to be normalized.
        min_val (float):
            Minimum value for the new range. Default: `0.0`.
        max_val (float):
            Maximum value for the new range. Default: `1.0`.
        eps (float):
            Float number to avoid zero division. Default: `1e-6`.

    Returns:
        output (Tensor[..., C, H, W]):
            Normalized tensor image with same shape.

    Example:
        >>> x      = torch.rand(1, 5, 3, 3)
        >>> x_norm = normalize_min_max(image, min_val=-1., max_val=1.)
        >>> x_norm.min()
        image(-1.)
        >>> x_norm.max()
        image(1.0000)
    """
    if not isinstance(image, Tensor):
        raise TypeError(
            f"`image` should be a `Tensor`. But got: {type(image)}."
        )
    if not isinstance(min_val, float):
        raise TypeError(
            f"`min_val` should be a `float`. But got: {type(min_val)}."
        )
    if not isinstance(max_val, float):
        raise TypeError(
            f"`max_val` should be a `float`. But got: {type(max_val)}."
        )
    if image.ndim < 3:
        raise ValueError(
            f"`image.ndim` must be >= 3. But got: {image.shape}."
        )

    shape = image.shape
    B, C  = shape[0], shape[1]

    x_min = image.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max = image.view(B, C, -1).max(-1)[0].view(B, C, 1)

    output = ((max_val - min_val) * (image.view(B, C, -1) - x_min) /
              (x_max - x_min + eps) + min_val)
    return output.view(shape)


@dispatch(Tensor)
def normalize_naive(image: Tensor) -> Tensor:
    """Convert image from `torch.uint8` type and range [0, 255] to `torch.float`
    type and range of [0.0, 1.0].
    
    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
    
    Returns:
        output (Tensor[..., C, H, W]):
            Normalized image Tensor.
    """
    if abs(torch.max(image)) > 1.0:
        return image.to(torch.get_default_dtype()).div(255.0)
    else:
        return image.to(torch.get_default_dtype())
    

@dispatch(list)
def normalize_naive(image: list) -> list:
    """Convert a list of images from `torch.uint8` type and range [0, 255]
    to `torch.float` type and range of [0.0, 1.0].
    
    Args:
        image (list[Tensor[..., C, H, W]]):
            List of image Tensor.
    
    Returns:
        output (list[Tensor[..., C, H, W]]):
            Normalized list of image Tensors.
    """
    if all(isinstance(i, Tensor) and i.ndim == 3 for i in image):
        image = normalize_naive(torch.stack(image))
        return list(image)
    elif all(isinstance(i, Tensor) and i.ndim == 4 for i in image):
        image = [normalize_naive(i) for i in image]
        return image
    else:
        raise TypeError(
            f"`image` must be a `list` of `Tensor`. But got: {type(image)}."
        )


@dispatch(tuple)
def normalize_naive(image: tuple) -> tuple:
    """Convert a tuple of images from `torch.uint8` type and range [0, 255]
    to `torch.float` type and range of [0.0, 1.0].
    
    Args:
        image (tuple[Tensor[..., C, H, W]]):
            Tuple of image Tensor.
    
    Returns:
        output (tuple[Tensor[..., C, H, W]]):
            Normalized tuple of image Tensors.
    """
    return tuple(normalize_naive(list(image)))


@dispatch(dict)
def normalize_naive(image: dict) -> dict:
    """Convert a dict of images from `torch.uint8` type and range [0, 255]
        to `torch.float` type and range of [0.0, 1.0].

        Args:
            image (dict):
                Dict of image Tensor.

        Returns:
            output (dict):
                Normalized dict of image Tensors.
        """
    if not all(isinstance(v, (Tensor, list, tuple)) for k, v in image.items()):
        raise ValueError(
            f"`image` must be a `dict` of `Tensor`, `list`, or `tuple`."
        )
    for k, v in image.items():
        image[k] = normalize_naive(v)
    return image


def pad_image(
    image   : Tensor,
    pad_size: Int2Or3T,
    mode    : Union[PaddingMode, str, int] = PaddingMode.CONSTANT,
    value   : Union[FloatAnyT, None]       = 0.0,
) -> Tensor:
    """Pad image with `value`.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be padded.
        pad_size (Int2Or3T[C, H, W]):
            Padded image size.
        mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `constant`.
        value (FloatAnyT, None):
            Fill value for `constant` padding. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Padded image.
    """
    if not 3 <= image.ndim <= 4:
        raise ValueError(
            f"Require 3 <= `image.ndim` <= 4. But got: {image.ndim}"
        )
    if not isinstance(mode, PaddingMode):
        mode = PaddingMode.from_value(value=mode)
    mode = mode.value
    if mode not in ("constant", "circular", "reflect", "replicate"):
        raise ValueError(
            f"`mode` must be one of ['constant', 'circular', 'reflect', "
            f"'replicate']. But got: {mode}."
        )
    
    h0, w0 = get_image_size(image)
    h1, w1 = to_size(pad_size)
    # Image size > pad size, do nothing
    if (h0 * w0) >= (h1 * w1):
        return image
    
    if value is None:
        value = 0
    pad_h = int(abs(h0 - h1) / 2)
    pad_w = int(abs(w0 - w1) / 2)

    if isinstance(image, Tensor):
        if is_channel_first(image):
            pad = (pad_w, pad_w, pad_h, pad_h)
        else:
            pad = (0, 0, pad_w, pad_w, pad_h, pad_h)
        return nn.functional.pad(input=image, pad=pad, mode=mode, value=value)
    return image


def resize(
    image        : Tensor,
    size         : Union[Int2Or3T, None]              = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.LINEAR,
    max_size     : Union[int, None]                   = None,
    antialias    : Union[bool, None]                  = None
) -> Tensor:
    """Resize an image. Adapted from:
    `torchvision.transforms.functional.resize()`
    
    Args:
        image (Tensor[..., C, H, W]):
            Image.
        size (Int2Or3T[C, H, W], None):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode, str, int):
            Interpolation method.
        max_size (int, None):
            Default: `None`.
        antialias (bool, None):
            Default: `None`.
            
    Returns:
        resize (Tensor[..., C, H, W]):
            Resized image.
    """
    if not isinstance(image, Tensor):
        raise TypeError(
            f"`image` must be a `Tensor` or `PIL Image`. "
            f"But got: {type(image)}."
        )
    
    if size is None:
        return image
    size = to_size(size)  # H, W
    
    if not isinstance(interpolation, InterpolationMode):
        interpolation = InterpolationMode.from_value(interpolation)
    if interpolation is InterpolationMode.LINEAR:
            interpolation = InterpolationMode.BILINEAR
    
    return F_t.resize(
        img           = image,
        size          = size,  # H, W
        interpolation = interpolation.value,
        max_size      = max_size,
        antialias     = antialias
    )


def rotate(
    image        : Tensor,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors Tensor[..., C, H, W].
    
    Args:
        image (Tensor[.., C, H, W]):
            Image to be rotated.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
            input image. Default: `True`.
            Note that the `keep_shape` flag assumes rotation around the center
            and no translation.
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (Tensor[..., C, H, W]):
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
    else:
        raise ValueError(
            f"`image` must be a `Tensor` or `PIL.Image`. "
            f"But got: {type(image)}."
        )


def rotate_horizontal_flip(
    image        : Tensor,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Rotate a tensor image or a batch of tensor images and then horizontally
    flip.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (Tensor[..., C, H, W]):
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
    return F.hflip(image)


def rotate_image_box(
    image        : Tensor,
    box          : Tensor,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0
) -> tuple[Tensor, Tensor]:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors Tensor[..., C, H, W].
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be rotated.
        box (Tensor[N, 4]):
            Bounding boxes to be rotated.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Rotated image.
        box (Tensor[N, 4]):
            Rotated bounding boxes.
    """
    if box.ndim != 2:
        raise ValueError(
            f"`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
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
    image        : Tensor,
    angle        : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Rotate a tensor image or a batch of tensor images and then vertically
    flip.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be rotated and flipped.
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
   
    Returns:
        image (Tensor[..., C, H, W]):
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
    return F.vflip(image)


def shear(
    image        : Tensor,
    magnitude    : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Shear image.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    box          : Tensor,
    magnitude    : FloatAnyT,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0
) -> tuple[Tensor, Tensor]:
    """Rotate a tensor image or a batch of tensor images. Input must be a
    tensor of shape [C, H, W] or a batch of tensors Tensor[..., C, H, W].
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be rotated.
        box (Tensor[N, 4]):
            Bounding boxes to be rotated.
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Rotated image.
        box (Tensor[N, 4]):
            Rotated bounding boxes.
    """
    if box.ndim != 2:
        raise ValueError(
            f"R`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
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


@dispatch(Tensor, keep_dims=bool)
def to_channel_first(image: Tensor, keep_dims: bool = True) -> Tensor:
    """Convert image to channel first format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
    
    Returns:
        image (np.ndarray):
            Image Tensor in channel first format.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image     = image.unsqueeze(0)
    elif image.ndim == 3:
        image     = image.permute(2, 0, 1)
    elif image.ndim == 4:
        image     = image.permute(0, 3, 1, 2)
        keep_dims = True
    elif image.ndim == 5:
        image     = image.permute(0, 1, 4, 2, 3)
        keep_dims = True
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image.unsqueeze(0) if not keep_dims else image


@dispatch(np.ndarray, keep_dims=bool)
def to_channel_first(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
    """Convert image to channel first format.
    
    Args:
        image (np.ndarray):
            Image array of arbitrary channel format.
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        
    Returns:
        image (np.ndarray):
            Image array in channel first format.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image    = np.expand_dims(image, 0)
    elif image.ndim == 3:
        image    = np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        image    = np.transpose(image, (0, 3, 1, 2))
        keep_dims = True
    elif image.ndim == 5:
        image    = np.transpose(image, (0, 1, 4, 2, 3))
        keep_dims = True
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return np.expand_dims(image, 0) if not keep_dims else image


@dispatch(Tensor, keep_dims=bool)
def to_channel_last(image: Tensor, keep_dims: bool = True) -> Tensor:
    """Convert image to channel last format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
    
    Returns:
        image (np.ndarray):
            Image Tensor in channel last format.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = image.permute(1, 2, 0)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        image = image.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = image.permute(0, 1, 3, 4, 2)
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image
    

@dispatch(np.ndarray, keep_dims=bool)
def to_channel_last(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
    """Convert image to channel last format.
    
    Args:
        image (np.ndarray):
            Image array of arbitrary channel format.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
            
    Returns:
        image (np.ndarray):
            Image array in channel last format.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        image = np.transpose(image, (0, 2, 3, 1))
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = np.transpose(image, (0, 1, 3, 4, 2))
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image


def to_image(
    input      : Tensor,
    keep_dims  : bool = True,
    denormalize: bool = False
) -> np.ndarray:
    """Converts a PyTorch Tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        input (Tensor):
            Image arbitrary shape.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
        
    Returns:
        image (np.ndarray):
            Image of the form [H, W], [H, W, C] or [..., H, W, C].
    """
    if not torch.is_tensor(input):
        error_console.log(f"Input type is not a Tensor. Got: {type(input)}.")
        return input
    if not 2 <= input.ndim <= 4:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 4. But got: {input.ndim}."
        )

    image = input.cpu().detach().numpy()
    
    # NOTE: Channel last format
    image = to_channel_last(image, keep_dims=keep_dims)
    
    # NOTE: Denormalize
    if denormalize:
        image = denormalize_naive(image)
        
    return image.astype(np.uint8)


def to_pil_image(image: TensorOrArray) -> PIL.Image:
    """Convert image from `np.ndarray` or `Tensor` to PIL image."""
    if torch.is_tensor(image):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return F.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
    image    : Union[Tensor, np.ndarray, PIL.Image],
    keep_dims: bool = True,
    normalize: bool = False,
) -> Tensor:
    """Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.
    
    Args:
        image (Tensor, np.ndarray, PIL.Image):
            Image array or PIL.Image in [H, W, C], [H, W] or [..., H, W, C].
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    
    Returns:
        img (Tensor):
            Image Tensor.
    """
    if not (F._is_numpy(image) or torch.is_tensor(image)
            or F._is_pil_image(image)):
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image. "
            f"But got: {type(image)}."
        )
    
    if ((F._is_numpy(image) or torch.is_tensor(image))
        and not (2 <= image.ndim <= 4)):
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 4. But got: {image.ndim}."
        )

    # img = image
    img = deepcopy(image)
    
    # NOTE: Handle PIL Image
    if F._is_pil_image(img):
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = np.array(img, mode_to_nptype.get(img.mode, np.uint8), copy=True)
        if image.mode == "1":
            img = 255 * img
    
    # NOTE: Handle numpy array
    if F._is_numpy(img):
        img = torch.from_numpy(img).contiguous()
    
    # NOTE: Channel first format
    img = to_channel_first(img, keep_dims=keep_dims)
   
    # NOTE: Normalize
    if normalize:
        img = normalize_naive(img)
    
    # NOTE: Convert type
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=torch.get_default_dtype())
    
    # NOTE: Place in memory
    img = img.contiguous()
    return img


def translate(
    image        : Tensor,
    magnitude    : Int2T,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Translate image in vertical and horizontal direction.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (Int2T):
            Horizontal and vertical translations (post-rotation translation).
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    box          : Tensor,
    magnitude    : Int2T,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int, int]  = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0
) -> tuple[Tensor, Tensor]:
    """Translate the image and bounding box with the given magnitude.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor[..., C, H, W]):
            Image to be translated.
        box (Tensor[N, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (Int2T):
            Horizontal and vertical translations (post-rotation translation).
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Translated image with the shape as the specified size.
        box (Tensor[N, 4]):
            Translated boxes.
    """
    if box.ndim != 2:
        raise ValueError(
            f"R`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
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
    image: Tensor, box: Tensor = ()
) -> tuple[Tensor, Tensor]:
    """Vertically flip images and bounding boxes.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image.
        box (Tensor[N, 4):
            Box.
    
    Returns:
        image (Tensor[..., C, H, W]):
            Flipped image.
        box (Tensor[N, 4):
            Flipped box.
    """
    if box.ndim != 2:
        raise ValueError(
            f"R`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
    center = get_image_center4(image)
    return F.vflip(image), vflip_box(box, center)
    

def vertical_shear(
    image        : Tensor,
    magnitude    : float,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Shear image vertically.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (int):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
) -> Tensor:
    """Translate image in vertical direction.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to transform.
        magnitude (int):
            Vertical translation (post-rotation translation)
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.

    Returns:
        image (Tensor[..., C, H, W]):
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
    image        : Tensor,
    box          : Tensor,
    magnitude    : int,
    center       : Union[ListOrTuple2T[int], None]    = None,
    interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
    keep_shape   : bool                               = True,
    pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
    fill         : Union[FloatAnyT, None]             = None,
    drop_ratio   : float                              = 0.0
) -> tuple[Tensor, Tensor]:
    """Translate the image and bounding box in vertical direction.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/
        
    Args:
        image (Tensor[..., C, H, W]):
            Image to be translated.
        box (Tensor[N, 4]):
            Box to be translated. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        magnitude (int):
            Vertical translation (post-rotation translation).
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Translated image with the shape as the specified size.
        box (Tensor[N, 4]):
            Translated boxes.
    """
    if box.ndim != 2:
        raise ValueError(
            f"R`box` must be a 2D `Tensor`. But got: {image.ndim}."
        )
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

@TRANSFORMS.register(name="add_weighted")
class AddWeighted(Transform):
    """Calculate the weighted sum of two Tensors.
    
    Function calculates the weighted sum of two Tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        alpha (float):
            Weight of the image1 elements.
        beta (float):
            Weight of the image2 elements.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return add_weighted(
            image1 = input,
            alpha  = self.alpha,
            image2 = target,
            beta   = self.beta,
            gamma  = self.gamma
        )
      

@TRANSFORMS.register(name="affine")
class Affine(Transform):
    """Apply affine transformation on the image keeping image center invariant.
    
    Args:
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
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
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
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.angle         = angle
        self.translate     = translate
        self.scale         = scale
        self.shear         = shear
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            affine(
                image         = input,
                angle         = self.angle,
                translate     = self.translate,
                scale         = self.scale,
                shear         = self.shear,
                center        = self.center,
                interpolation = self.interpolation,
                fill          = self.fill,
            ), \
            affine(
                image         = target,
                angle         = self.angle,
                translate     = self.translate,
                scale         = self.scale,
                shear         = self.shear,
                center        = self.center,
                interpolation = self.interpolation,
                fill          = self.fill,
            ) if target is not None else None
        

@TRANSFORMS.register(name="add_weighted")
class Blend(Transform):
    """Blends 2 images together.

    Args:
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.
    """

    # MARK: Magic Functions
    
    def __init__(self, alpha: float, gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(self, input : Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return blend(
            image1 = input,
            image2 = target,
            alpha  = self.alpha,
            gamma  = self.gamma,
        )
    

@TRANSFORMS.register(name="center_crop")
class CenterCrop(Transform):
    """Crops the given image at the center. If the image is Tensor, it is
    expected to have [..., H, W] shape, where ... means an arbitrary number of
    leading dimensions. If image size is smaller than output size along any
    edge, image is padded with 0 and then cropped.

    Args:
        size (Int2T):
            Desired output size of the crop. If size is an int instead of
            sequence like (h, w), a square crop (size, size) is made.
            If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
    """

    # MARK: Magic Functions
    
    def __init__(self, size: Int2T, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = to_size(size=size)

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return F.center_crop(img=input,  output_size=self.size), \
               F.center_crop(img=target, output_size=self.size) \
                    if target is not None else None
    
    
@TRANSFORMS.register(name="crop")
class Crop(Transform):
    """Crop image at specified location and output size. If image size is
    smaller than output size along any edge, image is padded with 0 and then
    cropped.

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

    # MARK: Magic Functions
    
    def __init__(
        self, top: int, left: int, height: int, width: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.top    = top
        self.left   = left
        self.height = height
        self.width  = width

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            F.crop(
                img    = input,
                top    = self.top,
                left   = self.left,
                height = self.height,
                width  = self.width
            ), \
            F.crop(
                img    = target,
                top    = self.top,
                left   = self.left,
                height = self.height,
                width  = self.width
            ) if target is not None else None


@TRANSFORMS.register(name="denormalize")
class Denormalize(Transform):
    """Denormalize an image Tensor with mean and standard deviation.
 
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        mean (Tensor[..., C, H, W], float):
            Mean for each channel.
        std (Tensor[..., C, H, W], float):
            Standard deviations for each channel.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        mean: Union[Tensor, float],
        std : Union[Tensor, float],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mean = mean
        self.std  = std
     
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return denormalize(image=input,  mean=self.mean, std=self.std), \
               denormalize(image=target, mean=self.mean, std=self.std) \
                    if target is not None else None


@TRANSFORMS.register(name="hflip")
@TRANSFORMS.register(name="horizontal_flip")
class HorizontalFlip(Transform):
    """Horizontally flip image."""

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return F.hflip(img=input), \
               F.hflip(img=target) if target is not None else None
    

@TRANSFORMS.register(name="hflip_image_box")
@TRANSFORMS.register(name="horizontal_flip_image_box")
class HorizontalFlipImageBox(Transform):
    """Horizontally flip image and bounding box."""
    
    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(
        self, input : Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return horizontal_flip_image_box(image=input, box=target)
   
   
@TRANSFORMS.register(name="hshear")
@TRANSFORMS.register(name="horizontal_shear")
class HorizontalShear(Transform):
    """Horizontally shear image.
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : float,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            horizontal_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ), \
            horizontal_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ) if target is not None else None


@TRANSFORMS.register(name="htranslate")
@TRANSFORMS.register(name="horizontal_translate")
class HorizontalTranslate(Transform):
    """Horizontally translate image.
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
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
        self,
        input : Tensor,
        target: Tensor = (),
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            horizontal_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ), \
            horizontal_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ) if target is not None else None
      

@TRANSFORMS.register(name="htranslate_image_box")
@TRANSFORMS.register(name="horizontal_translate_image_box")
class HorizontalTranslateImageBox(Transform):
    """Horizontally translate image and bounding box.
    
    Args:
        magnitude (int):
            Horizontal translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
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
        self, input: Tensor, target: Tensor = (), *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return horizontal_translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="resize")
class Resize(Transform):
    """Resize image to the given size.

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
        interpolation (InterpolationMode, str, int):
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

    # MARK: Magic Functions
    
    def __init__(
        self,
        size         : Union[Int3T, None],
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        max_size     : Union[int, None]                   = None,
        antialias    : Union[bool, None]                  = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.size          = size
        self.interpolation = interpolation
        self.max_size      = max_size
        self.antialias     = antialias

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            resize(
                image         = input,
                size          = self.size,
                interpolation = self.interpolation,
                max_size      = self.max_size,
                antialias     = self.antialias,
            ), \
            resize(
                image         = target,
                size          = self.size,
                interpolation = self.interpolation,
                max_size      = self.max_size,
                antialias     = self.antialias,
            ) if target is not None else None
    

@TRANSFORMS.register(name="resized_crop")
class ResizedCrop(Transform):
    """Resize and crop image to the given size.
    
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
        interpolation (InterpolationMode, str, int):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            F.resized_crop(
                img           = input,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation
            ), \
            F.resized_crop(
                img           = target,
                top           = self.top,
                left          = self.left,
                height        = self.height,
                width         = self.width,
                size          = self.size,
                interpolation = self.interpolation
            ) if target is not None else None


@TRANSFORMS.register(name="rotate")
class Rotate(Transform):
    """Rotate image.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            rotate(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ), \
            rotate(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ) if target is not None else None


@TRANSFORMS.register(name="rotate_hflip")
@TRANSFORMS.register(name="rotate_horizontal_flip")
class RotateHorizontalFlip(Transform):
    """Horizontally flip image.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            rotate_horizontal_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ), \
            rotate_horizontal_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ) if target is not None else None
    

@TRANSFORMS.register(name="rotate_vflip")
@TRANSFORMS.register(name="rotate_vertical_flip")
class RotateVerticalFlip(Transform):
    """Rotate and flip image.
    
    Args:
        angle (float):
            Angle to rotate the image.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        angle        : float,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.angle         = angle
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            rotate_vertical_flip(
                image         = input,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ), \
            rotate_vertical_flip(
                image         = target,
                angle         = self.angle,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ) if target is not None else None
    

@TRANSFORMS.register(name="shear")
class Shear(Transform):
    """Shear image.
    
    Args:
        magnitude (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : list[float],
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ), \
            shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ) if target is not None else None


@TRANSFORMS.register(name="to_image")
class ToImage(Transform):
    """Converts a PyTorch Tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        keep_dims  : bool = True,
        denormalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keep_dims   = keep_dims
        self.denormalize = denormalize
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            to_image(
                input       = input,
                keep_dims   = self.keep_dims,
                denormalize = self.denormalize
            ), \
            to_image(
                input       = target,
                keep_dims   = self.keep_dims,
                denormalize = self.denormalize
            ) if target is not None else None
    

@TRANSFORMS.register(name="to_tensor")
class ToTensor(Transform):
    """Convert a `PIL Image` or `np.ndarray` image to a 4D tensor.
    
    Args:
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        keep_dims: bool = False,
        normalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keep_dims = keep_dims
        self.normalize = normalize
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            to_tensor(
                image     = input,
                keep_dims = self.keep_dims,
                normalize = self.normalize
            ), \
            to_tensor(
                image     = input,
                keep_dims = self.keep_dims,
                normalize = self.normalize
            ) if target is not None else None
    

@TRANSFORMS.register(name="translate")
class Translate(Transform):
    """Translate image.
    
    Args:
        magnitude (Int2T):
            Horizontal and vertical translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : Int2T,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ), \
            translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ) if target is not None else None


@TRANSFORMS.register(name="translate_image_box")
class TranslateImageBox(Transform):
    """Translate image and bounding box.
    
    Args:
        magnitude (Int2T):
            Horizontal and vertical translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : Int2T,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return translate_image_box(
            image         = input,
            box           = target,
            magnitude     = self.magnitude,
            center        = self.center,
            interpolation = self.interpolation,
            keep_shape    = self.keep_shape,
            pad_mode      = self.pad_mode,
            fill          = self.fill,
        )


@TRANSFORMS.register(name="vflip")
@TRANSFORMS.register(name="vertical_flip")
class VerticalFlip(Transform):
    """Vertically flip image."""
  
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return F.vflip(img=input), \
               F.vflip(img=target) if target is not None else None
    
    
@TRANSFORMS.register(name="vflip_image_box")
@TRANSFORMS.register(name="vertical_flip_image_box")
class VerticalFlipImageBox(Transform):
    """Vertically flip image and bounding box."""
    
    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return vertical_flip_image_box(image=input, box=target)


@TRANSFORMS.register(name="yshear")
@TRANSFORMS.register(name="vertical_shear")
class VerticalShear(Transform):
    """Vertically shear image.
    
    Args:
        magnitude (float):
            Shear angle value in degrees between -180 to 180, clockwise
            direction.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : float,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            vertical_shear(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ), \
            vertical_shear(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill
            ) if target is not None else None


@TRANSFORMS.register(name="vtranslate")
@TRANSFORMS.register(name="vertical_translate")
class VerticalTranslate(Transform):
    """Vertically translate image.
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            vertical_translate(
                image         = input,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ), \
            vertical_translate(
                image         = target,
                magnitude     = self.magnitude,
                center        = self.center,
                interpolation = self.interpolation,
                keep_shape    = self.keep_shape,
                pad_mode      = self.pad_mode,
                fill          = self.fill,
            ) if target is not None else None


@TRANSFORMS.register(name="vtranslate_image_box")
@TRANSFORMS.register(name="vertical_translate_image_box")
class VerticalTranslateImageBox(Transform):
    """Vertically translate image and bounding box.
    
    Args:
        magnitude (int):
            Vertical translation magnitude.
        center (ListOrTuple2T[int], None):
            Center of affine transformation. If `None`, use the center of the
            image. Default: `None`.
        interpolation (InterpolationMode, str, int):
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
        pad_mode (PaddingMode, str, int):
            One of the padding modes defined in `PaddingMode`.
            Default: `PaddingMode.CONSTANT`.
        fill (FloatAnyT, None):
            Pixel fill value for the area outside the transformed image.
            If given a number, the value is used for all bands respectively.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        magnitude    : int,
        center       : Union[ListOrTuple2T[int], None]    = None,
        interpolation: Union[InterpolationMode, str, int] = InterpolationMode.BILINEAR,
        keep_shape   : bool                               = True,
        pad_mode     : Union[PaddingMode, str, int]       = PaddingMode.CONSTANT,
        fill         : Union[FloatAnyT, None]             = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.magnitude     = magnitude
        self.center        = center
        self.interpolation = interpolation
        self.keep_shape    = keep_shape
        self.pad_mode      = pad_mode
        self.fill          = fill
    
    # MARK: Forward Pass

    # noinspection PyMethodOverriding
    def forward(
        self, input: Tensor, target: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return vertical_translate_image_box(
            image         = input,
            box           = target,
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
