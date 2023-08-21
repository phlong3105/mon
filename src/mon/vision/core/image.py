#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the basic functionalities of image data.
"""

from __future__ import annotations

__all__ = [
    "add_weighted", "blend", "check_image_size", "denormalize_image",
    "denormalize_image_mean_std", "get_hw", "get_image_center",
    "get_image_center4", "get_image_num_channels", "get_image_shape",
    "get_image_size", "is_channel_first_image", "is_channel_last_image",
    "is_color_image", "is_gray_image", "is_image", "is_integer_image",
    "is_normalized_image", "is_one_hot_image", "normalize_image",
    "normalize_image_by_range", "normalize_image_mean_std", "to_3d_image",
    "to_4d_image", "to_5d_image", "to_channel_first_image",
    "to_channel_last_image", "to_image_nparray", "to_image_tensor",
    "to_list_of_3d_image",
]

import copy
import functools
from typing import Any

import numpy as np
import torch

from mon import nn
from mon.core import error_console, math


# region Assert

def is_channel_first_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format. We assume
    that if the first dimension is the smallest.
    """
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{input.ndim}."
        )
    if input.ndim == 5:
        _, _, s2, s3, s4 = list(input.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif input.ndim == 4:
        _, s1, s2, s3 = list(input.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif input.ndim == 3:
        s0, s1, s2 = list(input.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    return False


def is_channel_last_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format."""
    return not is_channel_first_image(input=input)


def is_color_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ```True``` if an image is a color image. It is assumed that the
    image has ``3`` or ``4`` channels.
    """
    if get_image_num_channels(input=input) in [3, 4]:
        return True
    return False


def is_gray_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a gray image. It is assumed that the
    image has ``1`` channel.
    """
    if get_image_num_channels(input=input) in [1] or len(input.shape) == 2:
        return True
    return False


def is_color_or_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ```True``` if an image is a color or gray image.
    """
    return is_color_image(input) or is_gray_image(input)


def is_image(input: torch.Tensor, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly :math:`[0, 1]` for
    :class:`float` or :math:`[0, 2 ** bits]` for :class:`int`.

    Args:
        input: Image tensor to evaluate.
        bits: The image bits. The default checks if given :class:`int` input
            image is an 8-bit image :math:`[0-255]` or not.

    Raises:
        TypeException: if all the input tensor has not 1) a shape
        :math:`[3, H, W]`, 2) :math:`[0, 1]` for :class:`float` or
        :math:`[0, 255]` for :class:`int`, 3) and raises is ``True``.
    
    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> is_image(img)
        True
    """
    res = is_color_or_image(input)
    if not res:
        return False
    if input.dtype in [torch.float16, torch.float32, torch.float64] \
        and (input.min() < 0.0 or input.max() > 1.0):
        return False
    elif input.min() < 0 or input.max() > 2**bits - 1:
        return False
    return True


def is_integer_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is integer-encoded."""
    c = get_image_num_channels(input=input)
    if c == 1:
        return True
    return False


def is_normalized_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is normalized."""
    if isinstance(input, torch.Tensor):
        return abs(torch.max(input)) <= 1.0
    elif isinstance(input, np.ndarray):
        return abs(np.amax(input)) <= 1.0
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, "
            f"but got {type(input)}."
        )


def is_one_hot_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is one-hot encoded."""
    c = get_image_num_channels(input=input)
    if c > 1:
        return True
    return False


def check_image_size(size: list[int], stride: int = 32) -> int:
    """If the input :param:`size` isn't a multiple of the :param:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: An image's size.
        stride: The stride of a network. Default: ``32``.
    
    Returns:
        A new size of the image.
    """
    size     = get_hw(size=size)
    size     = size[0]
    new_size = math.make_divisible(size, divisor=int(stride))
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size

# endregion


# region Access

def get_image_num_channels(input: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        input: An image in channel-last or channel-first format.
    """
    if not 2 <= input.ndim <= 4:
        raise ValueError(
            f"image's number of dimensions must be between 2 and 4, but got "
            f"{input.ndim}."
        )
    if input.ndim == 4:
        if is_channel_first_image(input=input):
            _, c, h, w = list(input.shape)
        else:
            _, h, w, c = list(input.shape)
    elif input.ndim == 3:
        if is_channel_first_image(input=input):
            c, h, w = list(input.shape)
        else:
            h, w, c = list(input.shape)
    else:
        c = 1
    return c


def get_image_center(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as :math:`(x=h/2, y=w/2)`.
    
    Args:
        input: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(input=input)
    if isinstance(input, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(input, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, "
            f"but got {type(input)}."
        )


def get_image_center4(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    :math:`(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        input: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(input=input)
    if isinstance(input, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(input, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, "
            f"but got {type(input)}."
        )


def get_image_size(input: torch.Tensor | np.ndarray) -> list[int]:
    """Return height and width value of an image.
    
    Args:
        input: An image.
    """
    if is_channel_first_image(input=input):
        return [input.shape[-2], input.shape[-1]]
    else:
        return [input.shape[-3], input.shape[-2]]


def get_image_shape(input: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        input: An image.
    """
    if is_channel_first_image(input=input):
        return [input.shape[-2], input.shape[-1], input.shape[-3]]
    else:
        return [input.shape[-3], input.shape[-2], input.shape[-1]]


def get_hw(size: int | list[int]) -> list[int]:
    """Casts a size object to the standard :math:`[H, W]` format.

    Args:
        size: A size of an image, windows, or kernels, etc.
    
    Returns:
        A size in :math:`[H, W]` format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            if size[0] >= size[3]:
                size = size[0:2]
            else:
                size = size[1:3]
        elif len(size) == 1:
            size = [size[0], size[0]]
    elif isinstance(size, int | float):
        size = [size, size]
    return size

# endregion


# region Convert

def denormalize_image_mean_std(
    input: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Denormalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        input: An image in channel-first format.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: A scalar value to avoid zero divisions. Default: ``1e-6``.
        
    Returns:
        A denormalized image.
    """
    if not input.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.clone()
        input = input.to(dtype=torch.get_default_dtype()) \
            if not input.is_floating_point() else input
        shape  = input.shape
        device = input.devices
        dtype  = input.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=input.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=input.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=input.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=input.devices)
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1) if std_inv.ndim == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        input.sub_(mean_inv).div_(std_inv)
    elif isinstance(input, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(input)}."
        )
    return input


def normalize_image_mean_std(
    input: torch.Tensor | np.ndarray,
    mean : float | list[float] = [0.485, 0.456, 0.406],
    std  : float | list[float] = [0.229, 0.224, 0.225],
    eps  : float               = 1e-6,
) -> torch.Tensor | np.ndarray:
    """Normalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where :obj:`mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for ``n``
    channels.

    Args:
        input: An image in channel-first format.
        mean: A sequence of means for each channel.
            Default: ``[0.485, 0.456, 0.406]``.
        std: A sequence of standard deviations for each channel.
            Default: ``[0.229, 0.224, 0.225]``.
        eps: A scalar value to avoid zero divisions. Default: ``1e-6``.
        
    Returns:
        A normalized image.
    """
    if not input.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.clone()
        input = input.to(dtype=torch.get_default_dtype()) \
            if not input.is_floating_point() else input
        shape  = input.shape
        device = input.devices
        dtype  = input.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=input.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=input.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=input.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=input.devices)
        std += eps
        
        mean = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std  = std.view(-1, 1, 1) if std.ndim == 1 else std
        input.sub_(mean).div_(std)
    elif isinstance(input, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(input)}."
        )
    return input


def normalize_image_by_range(
    input  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range [:param:`min`, :param:`max`] to the
    [:param:`new_min`, :param:`new_max`].
    
    Args:
        input: An image.
        min: The current minimum pixel value of the image. Default: ``0.0``.
        max: The current maximum pixel value of the image. Default: ``255.0``.
        new_min: A new minimum pixel value of the image. Default: ``0.0``.
        new_max: A new minimum pixel value of the image. Default: ``1.0``.
        
    Returns:
        A normalized image.
    """
    if not input.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {input.ndim}."
        )
    # if is_normalized_image(image=image):
    #     return image
    if isinstance(input, torch.Tensor):
        input = input.clone()
        input = input.to(dtype=torch.get_default_dtype()) \
            if not input.is_floating_point() else input
        ratio = (new_max - new_min) / (max - min)
        input = (input - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(input, np.ndarray):
        input = copy.deepcopy(input)
        input = input.astype(np.float32)
        ratio = (new_max - new_min) / (max - min)
        input = (input - min) * ratio + new_min
        # image = np.cip(image, new_min, new_max)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(input)}."
        )
    return input


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


def to_3d_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D or 4-D image to a 3-D.

    Args:
        image: An image in channel-first format.

    Return:
        A 3-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 4, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 1HW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 1HW
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_list_of_3d_image(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :class:`list` of 3-D images.
   
    Args:
        image: An image of arbitrary type.
        
    Return:
        A :class:`list` of 3-D images.
    """
    if isinstance(image, (torch.Tensor, np.ndarray)):
        if image.ndim == 3:
            image = [image]
        elif image.ndim == 4:
            image = list(image)
        else:
            raise ValueError
    elif isinstance(image, list | tuple):
        if not all(isinstance(i, (torch.Tensor, np.ndarray)) for i in image):
            raise ValueError
    return image


def to_4d_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, or 5-D image to a 4-D.

    Args:
        input: An image in channel-first format.

    Return:
        A 4-D image in channel-first format.
    """
    if not 2 <= input.ndim <= 5:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 5, but got "
            f"{input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        if input.ndim == 2:  # HW -> 11HW
            input = input.unsqueeze(dim=0)
            input = input.unsqueeze(dim=0)
        elif input.ndim == 3:  # CHW -> 1CHW
            input = input.unsqueeze(dim=0)
        elif input.ndim == 5 and input.shape[0] == 1:  # 1NCHW -> NCHW
            input = input.squeeze(dim=0)
    elif isinstance(input, np.ndarray):
        if input.ndim == 2:  # HW -> 11HW
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 3:  # CHW -> 1CHW
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 5 and input.shape[0] == 1:  # 1NCHW -> NHWC
            input = np.squeeze(input, axis=0)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(input)}."
        )
    return input


def to_5d_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, 4-D, or 6-D image to a 5-D.
    
    Args:
        input: An tensor in channel-first format.

    Return:
        A 5-D image in channel-first format.
    """
    if not 2 <= input.ndim <= 6:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 6, but got "
            f"{input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        if input.ndim == 2:  # HW -> 111HW
            input = input.unsqueeze(dim=0)
            input = input.unsqueeze(dim=0)
            input = input.unsqueeze(dim=0)
        elif input.ndim == 3:  # CHW -> 11CHW
            input = input.unsqueeze(dim=0)
            input = input.unsqueeze(dim=0)
        elif input.ndim == 4:  # NCHW -> 1NCHW
            input = input.unsqueeze(dim=0)
        elif input.ndim == 6 and input.shape[0] == 1:  # 1*NCHW -> *NCHW
            input = input.squeeze(dim=0)
    elif isinstance(input, np.ndarray):
        if input.ndim == 2:  # HW -> 111HW
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 3:  # HWC -> 11HWC
            input = np.expand_dims(input, axis=0)
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 4:  # BHWC -> 1BHWC
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 6 and input.shape[0] == 1:  # 1*BHWC -> *BHWC
            input = np.squeeze(input, axis=0)
    else:
        raise TypeError(
            f"x must be a np.ndarray or torch.Tensor, but got {type(input)}."
        )
    return input


def to_channel_first_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format.
    
    Args:
        input: An image in channel-last or channel-first format.
    
    Returns:
        An image in channel-first format.
    """
    if is_channel_first_image(input=input):
        return input
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.clone()
        if input.ndim == 3:
            input = input.permute(2, 0, 1)
        elif input.ndim == 4:
            input = input.permute(0, 3, 1, 2)
        elif input.ndim == 5:
            input = input.permute(0, 1, 4, 2, 3)
    elif isinstance(input, np.ndarray):
        input = copy.deepcopy(input)
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
        elif input.ndim == 4:
            input = np.transpose(input, (0, 3, 1, 2))
        elif input.ndim == 5:
            input = np.transpose(input, (0, 1, 4, 2, 3))
    else:
        raise TypeError(
            f"image must be torch.Tensor or a numpy.ndarray, "
            f"but got {type(input)}."
        )
    return input


def to_channel_last_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format.

    Args:
        input: An image in channel-last or channel-first format.

    Returns:
        A image in channel-last format.
    """
    if is_channel_last_image(input=input):
        return input
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.clone()
        if input.ndim == 3:
            input = input.permute(1, 2, 0)
        elif input.ndim == 4:
            input = input.permute(0, 2, 3, 1)
        elif input.ndim == 5:
            input = input.permute(0, 1, 3, 4, 2)
    elif isinstance(input, np.ndarray):
        input = copy.deepcopy(input)
        if input.ndim == 3:
            input = np.transpose(input, (1, 2, 0))
        elif input.ndim == 4:
            input = np.transpose(input, (0, 2, 3, 1))
        elif input.ndim == 5:
            input = np.transpose(input, (0, 1, 3, 4, 2))
    else:
        raise TypeError(
            f"image must be torch.Tensor or a numpy.ndarray, "
            f"but got {type(input)}."
        )
    return input


def to_image_nparray(
    input      : torch.Tensor | np.ndarray,
    keepdim    : bool = True,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :class:`numpy.ndarray`.
    
    Args:
        input: An image.
        keepdim: If `True`, keep the original shape. If ``False``, convert it to
            a 3-D shape. Default: ``True``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``True``.

    Returns:
        An :class:`numpy.ndarray` image.
    """
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.detach()
        input = input.cpu().numpy()
    input = denormalize_image(image=input).astype(np.uint) if denormalize else input
    input = to_channel_last_image(input=input)
    if not keepdim:
        input = to_3d_image(image=input)
    return input


def to_image_tensor(
    input    : torch.Tensor | np.ndarray,
    keepdim  : bool = True,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :class:`PIL.Image` or :class:`numpy.ndarray` to
    :class:`torch.Tensor`. Optionally, convert :param:`image` to channel-first
    format and normalize it.
    
    Args:
        input: An image in channel-last or channel-first format.
        keepdim: If ``True``, keep the original shape. If ``False``, convert it
            to a 4-D shape. Default: ``True``.
        normalize: If ``True``, normalize the image to :math:``[0, 1]``.
            Default: ``False``.
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        
    Returns:
        A :class:`torch.Tensor` image.
    """
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input).contiguous()
    elif isinstance(input, torch.Tensor):
        input = input.clone()
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, "
            f"but got {type(input)}."
        )
    input = to_channel_first_image(input=input)
    if not keepdim:
        input = to_4d_image(input=input)
    input = normalize_image(image=input) if normalize else input
    # Place in memory
    input = input.contiguous()
    if device is not None:
        device = nn.select_device(device=device) \
            if not isinstance(device, torch.device) else device
        input = input.to(device)
    return input

# endregion


# region Edit

def add_weighted(
    input1: torch.Tensor | np.ndarray,
    alpha : float,
    input2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        input1: The first image.
        alpha: The weight of the :param:`image1` elements.
        input2: The second image.
        beta: The weight of the :param:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if input1.shape != input2.shape:
        raise ValueError(
            f"The shape of x and y must be the same, but got "
            f"{input1.shape} and {input2.shape}."
        )
    bound  = 1.0 if input1.is_floating_point() else 255.0
    output = input1 * alpha + input2 * beta + gamma
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(input1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, "
            f"but got {type(output)}."
        )
    return output


def blend(
    input1: torch.Tensor | np.ndarray,
    input2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :param:`image1` * alpha + :param:`image2` * beta + gamma

    Args:
        input1: A source image.
        input2: A n overlay image that we want to blend on top of
            :param:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A blended image.
    """
    return add_weighted(
        input1 = input2,
        alpha  = alpha,
        input2 = input1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion
