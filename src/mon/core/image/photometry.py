#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Photometric Adjustments.

This module implements the basic functionalities of photometric operations. It
usually involves the manipulation of pixel values in an image such as light
intensity, brightness, or luminance.
"""

from __future__ import annotations

__all__ = [
    "add_noise",
    "adjust_gamma",
	"denormalize_image",
	"denormalize_image_mean_std",
	"normalize_image",
	"normalize_image_by_range",
	"normalize_image_mean_std",
]

import copy
import functools
from typing import Literal

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF


# region Gamma

def adjust_gamma(
    image: torch.Tensor | np.ndarray,
    gamma: float = 1.0,
    gain : float = 1.0
) -> torch.Tensor | np.ndarray:
    """Adjust gamma value in the image. Also known as Power Law Transform.
    
    Intensities in RGB mode are adjusted based on the following equation:
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        gamma: Non-negative real number, same as `gamma` in the equation.
            - :obj:`gamma` larger than ``1`` makes the shadows darker, while
            - :obj:`gamma` smaller than ``1`` makes dark regions lighter.
        gain: The constant multiplier.
        
    Returns:
        A gamma-corrected image.
    """
    if isinstance(image, torch.Tensor):
        return TF.adjust_gamma(img=image, gamma=gamma, gain=gain)
    elif isinstance(image, np.ndarray):
        # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values.
        inv_gamma = 1.0 / gamma
        table     = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
        table.astype("uint8")
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    
# endregion


# region Noise

def add_noise(
    image      : torch.Tensor,
    noise_level: int = 25,
    noise_type : Literal["gaussian", "poisson"] = "gaussian"
) -> torch.Tensor:
    """Add noise to an image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        noise_level: The noise level.
        noise_type: The type of noise to add. One of:
            - ``'gaussian'``
            - ``'poisson'``
            Default: ``"gaussian"``.
        
    Returns:
        A noisy image.
    """
    if noise_type == "gaussian":
        noisy = image + torch.normal(0, noise_level / 255, image.shape)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == "poisson":
        noisy = torch.poisson(noise_level * image) / noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noisy

# endregion


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
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0.0, 1.0]``.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-6``.
        
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
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0, 255]``.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-6``.
        
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
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0, 255]``.
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
