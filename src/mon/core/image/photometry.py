#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Photometric Adjustments.

This module implements the basic functionalities of photometric operations. It
usually involves the manipulation of pixel values in an image such as light
intensity, brightness, or luminance.
"""

from __future__ import annotations

__all__ = [
	"denormalize_image",
	"denormalize_image_mean_std",
	"normalize_image",
	"normalize_image_by_range",
	"normalize_image_mean_std",
]

import copy
import functools

import numpy as np
import torch


# region Normalize

def denormalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Denormalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image in channel-first format.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: A scalar value to avoid zero divisions. Default: ``1e-6``.
        
    Returns:
        A denormalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.device
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], dtype=dtype, device=device)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.device)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], dtype=dtype, device=device)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.device)
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1)  if std_inv.ndim  == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


def normalize_image_mean_std(
    image: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where :obj:`mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for ``n``
    channels.

    Args:
        image: An image in channel-first format.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: A scalar value to avoid zero divisions. Default: ``1e-6``.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.device
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.device)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.device)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.device)
        std += eps
        
        mean = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std  = std.view(-1, 1, 1)  if std.ndim  == 1 else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range ``[:obj:`min`, :obj:`max`]`` to the
    ``[:obj:`new_min`, :obj:`new_max`]``.
    
    Args:
        image: An image.
        min: The current minimum pixel value of the image. Default: ``0.0``.
        max: The current maximum pixel value of the image. Default: ``255.0``.
        new_min: A new minimum pixel value of the image. Default: ``0.0``.
        new_max: A new minimum pixel value of the image. Default: ``1.0``.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(f"`image`'s number of dimensions must be >= ``3``, "
                         f"but got {image.ndim}.")
    # if is_normalized_image(image=image):
    #     return image
    if isinstance(image, torch.Tensor):
        image = image.clone()
        # input = input.to(dtype=torch.get_default_dtype()) if not input.is_floating_point() else input
        image = image.to(dtype=torch.get_default_dtype())
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        image = image.astype(np.float32)
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = np.clip(image, new_min, new_max)
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


denormalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 1.0,
    new_min = 0.0,
    new_max = 255.0
)
normalize_image = functools.partial(
    normalize_image_by_range,
    min     = 0.0,
    max     = 255.0,
    new_min = 0.0,
    new_max = 1.0
)

# endregion
