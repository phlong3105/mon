#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformation on pixel intensity.
"""

from __future__ import annotations

import inspect
import math
import numbers
import sys
from typing import Any
from typing import Union

import torch
import torchvision.transforms.functional_tensor as F_t
from multipledispatch import dispatch
from torch import Tensor

from one.core import assert_float
from one.core import assert_number_in_range
from one.core import assert_positive_number
from one.core import assert_same_shape
from one.core import assert_sequence_of_length
from one.core import assert_tensor
from one.core import assert_tensor_of_atleast_ndim
from one.core import assert_tensor_of_channels
from one.core import assert_tensor_of_ndim_in_range
from one.core import Floats
from one.core import Transform
from one.core import TRANSFORMS
from one.vision.acquisition import get_image_shape
from one.vision.acquisition import get_num_channels
from one.vision.transformation.color import hsv_to_rgb
from one.vision.transformation.color import rgb_to_grayscale
from one.vision.transformation.color import rgb_to_hsv


# MARK: - Functional

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
    assert_tensor(image1)
    assert_tensor(image2)
    assert_same_shape(image1, image2)
    assert_float(alpha)
    assert_float(beta)
    assert_float(gamma)
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    output = output.clamp(0, bound).to(image1.dtype)
    return output


def adjust_brightness(image: Tensor, brightness_factor: float) -> Tensor:
    """Adjust brightness of an image.

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        brightness_factor (float):
            How much to adjust the brightness. Can be any non-negative number.
            0 gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        
    Returns:
        image (Tensor):
            Brightness adjusted image.
    """
    assert_positive_number(brightness_factor)
    assert_tensor_of_channels(image, [1, 3])
    return blend(
	    image1 = image,
	    alpha  = brightness_factor,
	    image2 = torch.zeros_like(image)
    )


def adjust_contrast(image: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of an image.

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        contrast_factor (float):
            How much to adjust the contrast. Can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2
            increases the contrast by a factor of 2.

    Returns:
        image (Tensor):
            Contrast adjusted image.
    """
    assert_positive_number(contrast_factor)
    assert_tensor_of_channels(image, [1, 3])
    c     = get_num_channels(image)
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    if c == 3:
        mean = torch.mean(
            rgb_to_grayscale(image=image).to(dtype),
            dim=(-3, -2, -1), keepdim=True
        )
    else:
        mean = torch.mean(image.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return blend(image1=image, alpha=contrast_factor, image2=mean)


def adjust_gamma(image: Tensor, gamma: float, gain: float = 1.0) -> Tensor:
    """Adjust gamma of an image.

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        gamma (float):
            How much to adjust the gamma. Can be any non-negative number.
            0 gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        gain (float):
            Default: `1.0`.
        
    Returns:
        result (Tensor):
            Gamma adjusted image.
    """
    assert_positive_number(gamma)
    assert_tensor_of_channels(image, [1, 3])

    result = image
    dtype  = image.dtype
    if not torch.is_floating_point(image):
        result = F_t.convert_image_dtype(result, torch.float32)
    result = (gain * result ** gamma).clamp(0, 1)
    result = F_t.convert_image_dtype(result, dtype)
    return result


def adjust_hue(image: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and cyclically
    shifting the intensities in the hue channel (H). The image is then
    converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        hue_factor (float):
            How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5
            and -0.5 give complete reversal of hue channel in HSV space in
            positive and negative direction respectively. 0 means no shift.
            Therefore, both -0.5 and 0.5 will give an image with complementary
            colors while 0 gives the original image.

    Returns:
        image (Tensor):
            Hue adjusted image.
    """
    assert_number_in_range(hue_factor, -0.5, 0.5)
    assert_tensor_of_channels(image, [1, 3])

    orig_dtype = image.dtype
    if image.dtype == torch.uint8:
        image  = image.to(dtype=torch.float32) / 255.0

    image       = rgb_to_hsv(image)
    h, s, v     = image.unbind(dim=-3)
    h           = (h + hue_factor) % 1.0
    image       = torch.stack((h, s, v), dim=-3)
    img_hue_adj = hsv_to_rgb(image)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(image: Tensor, saturation_factor: float) -> Tensor:
    """Adjust color saturation of an image.

    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        saturation_factor (float):
            How much to adjust the saturation. 0 will give a black and white
            image, 1 will give the original image while 2 will enhance the
            saturation by a factor of 2.

    Returns:
        image (Tensor):
             Saturation adjusted image.
    """
    assert_positive_number(saturation_factor)
    assert_tensor_of_channels(image, [1, 3])
    
    if get_num_channels(image) == 1:
        return image
    
    return blend(
        image1 = image,
        alpha  = saturation_factor,
        image2 = rgb_to_grayscale(image=image)
    )


def adjust_sharpness(image: Tensor, sharpness_factor: float) -> Tensor:
    """Adjust sharpness of an image.
    
    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        sharpness_factor (float):
            How much to adjust the sharpness. 0 will give a black and white
            image, 1 will give the original image while 2 will enhance the
            saturation by a factor of 2.
    
    Returns:
        image (Tensor):
             Sharpness adjusted image.
    """
    assert_positive_number(sharpness_factor)
    assert_tensor_of_channels(image, [1, 3])
    
    if image.size(-1) <= 2 or image.size(-2) <= 2:
        return image

    return blend(
        image1 = image,
        image2 = F_t._blurred_degenerate_image(img=image),
        alpha  = sharpness_factor,
    )


def autocontrast(image: Tensor) -> Tensor:
    """Maximize contrast of an image by remapping its pixels per channel so
    that the lowest becomes black and the lightest becomes white.
    
    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
    
    Returns:
        image (Tensor):
             Auto-contrast adjusted image.
    """
    assert_tensor_of_channels(image, [1, 3])
    assert_tensor_of_atleast_ndim(image, 3)

    bound = 1.0 if image.is_floating_point() else 255.0
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32

    minimum          = image.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum          = image.amax(dim=(-2, -1), keepdim=True).to(dtype)
    scale            = bound / (maximum - minimum)
    eq_idxs          = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs]   = 1

    return ((image - minimum) * scale).clamp(0, bound).to(image.dtype)


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


def denormalize(
    image  : Tensor,
    mean   : Union[Tensor, float],
    std    : Union[Tensor, float],
    inplace: bool = False,
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
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
            
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
    
    assert_tensor(image)
    assert_tensor(mean)
    assert_tensor(std)
    
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
    
    if not inplace:
        image = image.clone()
    
    image = (image.view(shape[0], shape[1], -1) * std) + mean
    image = image.view(shape)
    return image


@dispatch(Tensor)
def denormalize_naive(image: Tensor) -> Tensor:
    """Naively denormalize an image Tensor.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
     
    Returns:
        image (Tensor[..., C, H, W]):
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
        image (list[Tensor[..., C, H, W]]):
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
        image (tuple[Tensor[..., C, H, W]]):
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


def erase(
    image  : Tensor,
    i      : int,
    j      : int,
    h      : int,
    w      : int,
    v      : Tensor,
    inplace: bool = False
) -> Tensor:
    """Erase the input Tensor Image with given value.

    Args:
        image (Tensor[..., C, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
        i (int):
            i in (i,j) i.e coordinates of the upper left corner.
        j (int):
            j in (i,j) i.e coordinates of the upper left corner.
        h (int):
            Height of the erased region.
        w (int):
            Width of the erased region.
        v (Tensor):
            Erasing value.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.

    Returns:
        image (Tensor):
            Erased image.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not inplace:
        image = image.clone()
    image[..., i: i + h, j: j + w] = v
    return image


def equalize(image: Tensor) -> Tensor:
    """Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        image (Tensor[..., 1 or 3, H, W]):
            Image to be adjusted, where ... means it can have an arbitrary
            number of leading dimensions.
    
    Returns:
        image (Tensor):
             Equalized image.
    """
    assert_tensor_of_ndim_in_range(image, 3, 4)
    assert_tensor_of_channels(image, [1, 3])
    
    if image.dtype != torch.uint8:
        raise TypeError(
            f"Only `torch.uint8` image tensors are supported. "
            f"But got: {image.dtype}."
        )
    if image.ndim == 3:
        return F_t._equalize_single_image(image)

    return torch.stack([F_t._equalize_single_image(x) for x in image])


def invert(image: Tensor) -> Tensor:
    """Invert the colors of an RGB/grayscale image.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be transformed, where ... means it can have an arbitrary
            number of leading dimensions.
      
    Returns:
        image (Tensor[..., C, H, W]):
            Inverted image.
    """
    assert_tensor_of_channels(image, [1, 3])
    bound = torch.tensor(
        data   = 1 if image.is_floating_point() else 255,
        dtype  = image.dtype,
        device = image.device
    )
    return bound - image


def is_normalized(image: Tensor) -> Tensor:
    """Return `True` if the given image is normalized."""
    assert_tensor(image)
    return abs(torch.max(image)) <= 1.0


def is_integer_image(image: Tensor) -> bool:
    """Return `True` if the given image is integer-encoded."""
    assert_tensor(image)
    c = get_num_channels(image)
    if c == 1:
        return True
    return False


def is_one_hot_image(image: Tensor) -> bool:
    """Return `True` if the given image is one-hot encoded."""
    assert_tensor(image)
    c = get_num_channels(image)
    if c > 1:
        return True
    return False
   

def normalize(
    image  : Tensor,
    mean   : list[float],
    std    : list[float],
    inplace: bool = False
) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not
        mutates the input tensor.

    Args:
        image (Tensor):
            Float tensor image of size (C, H, W) or (B, C, H, W) to be
            normalized.
        mean (list[float]):
            Sequence of means for each channel.
        std (list[float]):
            Sequence of standard deviations for each channel.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.

    Returns:
        image (Tensor):
            Normalized Tensor image.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not image.is_floating_point():
        raise TypeError(
            f"Input tensor should be a float Tensor. Got {image.dtype}."
        )

    if not inplace:
        image = image.clone()

    dtype = image.dtype
    mean  = torch.as_tensor(mean, dtype=dtype, device=image.device)
    std   = torch.as_tensor(std,  dtype=dtype, device=image.device)
    if (std == 0).any():
        raise ValueError(
            f"`std` evaluated to zero after conversion to {dtype}, leading to "
            f"division by zero."
        )
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image.sub_(mean).div_(std)
    return image
    

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
    assert_tensor(image)
    assert_tensor_of_atleast_ndim(image, 3)
    assert_float(min_val)
    assert_float(max_val)

    shape  = image.shape
    B, C   = shape[0], shape[1]

    x_min  = image.view(B, C, -1).min(-1)[0].view(B, C, 1)
    x_max  = image.view(B, C, -1).max(-1)[0].view(B, C, 1)

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


def posterize(image: Tensor, bits: int) -> Tensor:
    """Posterize an image by reducing the number of bits for each color channel.
    
    Args:
        image (Tensor[..., C, H, W]):
            Image to be transformed, where ... means it can have an arbitrary
            number of leading dimensions.
        bit (int):
            Number of bits to keep for each channel (0-8).
        
    Returns:
        image (Tensor[..., C, H, W]):
            Posterized image.
    """
    assert_tensor_of_channels(image, [1, 3])
    if image.dtype != torch.uint8:
        raise TypeError(
            f"Only `torch.uint8` image tensors are supported. "
            f"But got: {image.dtype}"
        )
    mask = -int(2 ** (8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return image & mask


def solarize(image: Tensor, threshold: float) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        image (Tensor[..., C, H, W]):
            Image to be transformed, where ... means it can have an arbitrary
            number of leading dimensions.
        threshold (float):
            All pixels equal or above this value are inverted.
            
    Returns:
        image (Tensor[..., C, H, W]):
            Solarized image.
    """
    assert_tensor_of_channels(image, [1, 3])
    
    bound = 1 if image.is_floating_point() else 255
    if threshold > bound:
        raise TypeError("Threshold should be less than bound of img.")
    
    inverted_img = invert(image)
    return torch.where(image >= threshold, inverted_img, image)


# MARK: - Module

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
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        alpha: float,
        beta : float,
        gamma: float,
        p    : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
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


@TRANSFORMS.register(name="adjust_brightness")
class AdjustBrightness(Transform):
    """Adjust brightness of an image.

    Args:
        brightness_factor (float):
            How much to adjust the brightness. Can be any non-negative number.
            0 gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        brightness_factor: float,
        p                : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness_factor = brightness_factor
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_brightness(
                image             = input,
                brightness_factor = self.brightness_factor
            ), \
            adjust_brightness(
                image             = target,
                brightness_factor = self.brightness_factor
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_contrast")
class AdjustContrast(Transform):
    """Adjust contrast of an image.

    Args:
        contrast_factor (float):
            How much to adjust the contrast. Can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2
            increases the contrast by a factor of 2.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        contrast_factor: float,
        p              : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.contrast_factor = contrast_factor
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_contrast(
                image           = input,
                contrast_factor = self.contrast_factor
            ), \
            adjust_contrast(
                image           = target,
                contrast_factor = self.contrast_factor
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_gamma")
class AdjustGamma(Transform):
    """Adjust gamma of an image.

    Args:
        gamma (float):
            How much to adjust the gamma. Can be any non-negative number.
            0 gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.
        gain (float):
            Default: `1.0`.
       p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        gamma: float,
        gain : float              = 1.0,
        p    : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.gamma = gamma
        self.gain  = gain
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_gamma(
                image = input,
                gamma = self.gamma,
                gain  = self.gain
            ), \
            adjust_gamma(
                image = target,
                gamma = self.gamma,
                gain  = self.gain
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_hue")
class AdjustHue(Transform):
    """Adjust hue of an image.

    Args:
        hue_factor (float):
            How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5
            and -0.5 give complete reversal of hue channel in HSV space in
            positive and negative direction respectively. 0 means no shift.
            Therefore, both -0.5 and 0.5 will give an image with complementary
            colors while 0 gives the original image.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        hue_factor: float,
        p         : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.hue_factor = hue_factor
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_hue(
                image      = input,
                hue_factor = self.hue_factor
            ), \
            adjust_hue(
                image      = target,
                hue_factor = self.hue_factor
            ) if target is not None else None
        

@TRANSFORMS.register(name="adjust_saturation")
class AdjustSaturation(Transform):
    """Adjust color saturation of an image.

    Args:
        saturation_factor (float):
            How much to adjust the saturation. 0 will give a black and white
            image, 1 will give the original image while 2 will enhance the
            saturation by a factor of 2.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        saturation_factor: float,
        p                : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.saturation_factor = saturation_factor
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_saturation(
                image             = input,
                saturation_factor = self.saturation_factor
            ), \
            adjust_saturation(
                image             = target,
                saturation_factor = self.saturation_factor
            ) if target is not None else None
     

@TRANSFORMS.register(name="adjust_sharpness")
class AdjustSharpness(Transform):
    """Adjust color sharpness of an image.

    Args:
        sharpness_factor (float):
            How much to adjust the sharpness. 0 will give a black and white
            image, 1 will give the original image while 2 will enhance the
            saturation by a factor of 2.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        sharpness_factor: float,
        p               : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.sharpness_factor = sharpness_factor
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            adjust_sharpness(
                image            = input,
                sharpness_factor = self.sharpness_factor
            ), \
            adjust_sharpness(
                image            = target,
                sharpness_factor = self.sharpness_factor
            ) if target is not None else None


@TRANSFORMS.register(name="autocontrast")
class AutoContrast(Transform):
    """Maximize contrast of an image by remapping its pixels per channel so
    that the lowest becomes black and the lightest becomes white.
    
    Args:
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions

    def __init__(self, p: Union[float, None] = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    # MARK: Forward Pass

    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return autocontrast(image=input),  \
               autocontrast(image=target) if target is not None else None
          

@TRANSFORMS.register(name="add_weighted")
class Blend(Transform):
    """Blends 2 images together.

    Args:
        alpha (float):
            Alpha transparency of the overlay.
        gamma (float):
            Scalar added to each sum. Default: `0.0`.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        alpha: float,
        gamma: float,
        p    : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
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


@TRANSFORMS.register(name="color_jitter")
class ColorJitter(Transform):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    
    Args:
        brightness (Floats):
            How much to jitter brightness. `brightness_factor` is chosen
            uniformly from [max(0, 1 - brightness), 1 + brightness] or the
            given [min, max]. Should be non negative numbers. Default: `0.0`.
        contrast (Floats):
            How much to jitter contrast. `contrast_factor` is chosen uniformly
            from [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non-negative numbers. Default: `0.0`.
        saturation (Floats):
            How much to jitter saturation. `saturation_factor` is chosen
            uniformly from [max(0, 1 - saturation), 1 + saturation] or the given
            [min, max]. Should be non-negative numbers. Default: `0.0`.
        hue (Floats):
            How much to jitter hue. `hue_factor` is chosen uniformly from
            [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5
            or -0.5 <= min <= max <= 0.5. Default: `0.0`.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        brightness: Union[Floats, None] = 0.0,
        contrast  : Union[Floats, None] = 0.0,
        saturation: Union[Floats, None] = 0.0,
        hue       : Union[Floats, None] = 0.0,
        p         : Union[float,   None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast   = self._check_input(contrast,   "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue        = self._check_input(
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
    
    # MARK: Configure
    
    @torch.jit.unused
    def _check_input(
        self,
        value             : Any,
        name              : str,
        center            : int   = 1,
        bound             : tuple = (0, float("inf")),
        clip_first_on_zero: bool  = True
    ) -> Union[Floats, None]:
        if isinstance(value, numbers.Number):
            assert_positive_number(value)
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)):
            assert_sequence_of_length(value, 2)
            assert_number_in_range(value[0], bound[0], bound[1])
            assert_number_in_range(value[1], bound[0], bound[1])
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with "
                f"length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    @staticmethod
    def get_params(
        brightness: Union[Floats, None],
        contrast  : Union[Floats, None],
        saturation: Union[Floats, None],
        hue       : Union[Floats, None],
    ) -> tuple[
        Tensor,
        Union[float, None],
        Union[float, None],
        Union[float, None],
        Union[float, None]
    ]:
        """Get the parameters for the randomized transform to be applied on
        image.

        Args:
            brightness (Floats, None):
                The range from which the `brightness_factor` is chosen uniformly.
                Pass `None` to turn off the transformation.
            contrast (Floats, None):
                The range from which the `contrast_factor` is chosen uniformly.
                Pass `None` to turn off the transformation.
            saturation (Floats, None):
                The range from which the `saturation_factor` is chosen uniformly.
                Pass `None` to turn off the transformation.
            hue (Floats, None):
                The range from which the `hue_factor` is chosen uniformly.
                Pass `None` to turn off the transformation.

        Returns:
            (tuple):
                The parameters used to apply the randomized transform along
                with their random order.
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
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        fn_idx, \
        brightness_factor, contrast_factor, saturation_factor, hue_factor \
            = self.get_params(
                brightness = self.brightness,
                contrast   = self.contrast,
                saturation = self.saturation,
                hue        = self.hue
            )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                input  = adjust_brightness(input,  brightness_factor)
                target = adjust_brightness(target, brightness_factor) \
                    if target is not None else None
            elif fn_id == 1 and contrast_factor is not None:
                input  = adjust_contrast(input, contrast_factor)
                target = adjust_contrast(target, contrast_factor) \
                    if target is not None else None
            elif fn_id == 2 and saturation_factor is not None:
                input  = adjust_saturation(input,  saturation_factor)
                target = adjust_saturation(target, saturation_factor) \
                    if target is not None else None
            elif fn_id == 3 and hue_factor is not None:
                input  = adjust_hue(input,  hue_factor)
                target = adjust_hue(target, hue_factor) \
                    if target is not None else None

        return input, target


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
        p   : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
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


@TRANSFORMS.register(name="erase")
class Erase(Transform):
    """Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        i (int):
            i in (i,j) i.e coordinates of the upper left corner.
        j (int):
            j in (i,j) i.e coordinates of the upper left corner.
        h (int):
            Height of the erased region.
        w (int):
            Width of the erased region.
        v (Tensor):
            Erasing value.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        i      : int,
        j      : int,
        h      : int,
        w      : int,
        v      : Tensor,
        inplace: bool               = False,
        p      : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.i       = i
        self.j       = j
        self.h       = h
        self.w       = w
        self.v       = v
        self.inplace = inplace
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            erase(
                image   = input,
                i       = self.i,
                j       = self.j,
                h       = self.h,
                w       = self.w,
                v       = self.v,
                inplace = self.inplace,
            ), \
            erase(
                image   = target,
                i       = self.i,
                j       = self.j,
                h       = self.h,
                w       = self.w,
                v       = self.v,
                inplace = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="equalize")
class Equalize(Transform):
    """Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions

    def __init__(self, p: Union[float, None] = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return equalize(image=input), \
               equalize(image=target) if target is not None else None


@TRANSFORMS.register(name="invert")
class Invert(Transform):
    """Invert the colors of an RGB/grayscale image.
    
    Args:
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions

    def __init__(self, p: Union[float, None] = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)

    # MARK: Forward Pass
   
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return invert(image=input), \
               invert(image=target) if target is not None else None


@TRANSFORMS.register(name="normalize")
class Normalize(Transform):
    """Normalize a tensor image with mean and standard deviation.
 
    Args:
        mean (list[float]):
            Sequence of means for each channel.
        std (list[float]):
            Sequence of standard deviations for each channel.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        mean   : list[float],
        std    : list[float],
        inplace: bool               = False,
        p      : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.mean    = mean
        self.std     = std
        self.inplace = inplace
     
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            normalize(
                image   = input,
                mean    = self.mean,
                std     = self.std,
                inplace = self.inplacem
            ), \
            normalize(
                image   = target,
                mean    = self.mean,
                std     = self.std,
                inplace = self.inplacem
            ) \
                if target is not None else None


@TRANSFORMS.register(name="posterize")
class Posterize(Transform):
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        bits (int):
            Number of bits to keep for each channel (0-8).
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        bits: int,
        p   : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.bits = bits

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return posterize(image=input,  bits=self.bits), \
               posterize(image=target, bits=self.bits) \
                   if target is not None else None


@TRANSFORMS.register(name="random_erase")
class RandomErase(Transform):
    """Randomly selects a rectangle region in an image Tensor and erases its
    pixels.
    
    References:
        'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    
    Args:
        scale (Floats):
            Range of proportion of erased area against input image.
        ratio (Floats):
            Range of aspect ratio of erased area.
        value (int, float, str, tuple, list):
            Erasing value. Default is `0`. If a single int, it is used to erase
            all pixels. If a tuple of length 3, it is used to erase R, G, B
            channels respectively. If a str of `random`, erasing each pixel
            with random values.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        scale  : Floats                             = (0.02, 0.33),
        ratio  : Floats                             = (0.3, 3.3),
        value  : Union[int, float, str, tuple, list] = 0,
        inplace: bool                                = False,
        p      : Union[float, None]                  = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
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
    
    # MARK: Configure
    
    @staticmethod
    def get_params(
        image: Tensor,
        scale: Floats,
        ratio: Floats,
        value: Union[list[float], None] = None
    ) -> tuple[int, int, int, int, Tensor]:
        """Get parameters for `erase` for a random erasing.

        Args:
            image (Tensor):
                Tensor image to be erased.
            scale (Floats):
                Range of proportion of erased area against input image.
            ratio (Floats):
                Range of aspect ratio of erased area.
            value (list[float], None):
                Erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If `len(value)` is
                1, it is interpreted as a number, i.e. `value[0]`.

        Returns:
            (tuple):
                Params (i, j, h, w, v) to be passed to `erase` for random
                erasing.
        """
        img_c, img_h, img_w = get_image_shape(image)
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
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        
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
            erase(
                image   = input,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
                inplace = self.inplace,
            ), \
            erase(
                image   = target,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
                inplace = self.inplace,
            ) if target is not None else None
    

@TRANSFORMS.register(name="solarize")
class Solarize(Transform):
    """Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        threshold (float):
            All pixels equal or above this value are inverted.
        p (float):
            Probability of the image being adjusted. Default: `None` means 
            process as normal.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        threshold: float,
        p        : Union[float, None] = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.threshold = threshold

    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return solarize(image=input,  threshold=self.threshold), \
               solarize(image=target, threshold=self.threshold) \
                   if target is not None else None
    
    
# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
