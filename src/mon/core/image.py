#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the basic functionalities of image data."""

from __future__ import annotations

__all__ = [
    "check_image_size",
    "denormalize_image",
    "denormalize_image_mean_std",
    "get_channel",
    "get_first_channel",
    "get_image_center",
    "get_image_center4",
    "get_image_num_channels",
    "get_image_shape",
    "get_image_size",
    "get_last_channel",
    "is_channel_first_image",
    "is_channel_last_image",
    "is_color_image",
    "is_gray_image",
    "is_image",
    "is_integer_image",
    "is_normalized_image",
    "is_one_hot_image",
    "normalize_image",
    "normalize_image_by_range",
    "normalize_image_mean_std",
    "read_image",
    "to_3d_image",
    "to_4d_image",
    "to_5d_image",
    "to_channel_first_image",
    "to_channel_last_image",
    "to_image_nparray",
    "to_image_tensor",
    "to_list_of_3d_image",
    "write_image",
    "write_image_cv",
    "write_image_torch",
    "write_images_cv",
    "write_images_torch",
]

import copy
import functools
import multiprocessing
from typing import Any, Sequence

import cv2
import joblib
import numpy as np
import torch
import torchvision

from mon.core import pathlib, utils
from mon.core.rich import error_console


# region Assert

def is_channel_first_image(input: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format. We assume
    that if the first dimension is the smallest.
    """
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {input.ndim}."
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
    image has ones channel.
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
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
    size     = utils.parse_hw(size=size)
    size     = size[0]
    new_size = utils.make_divisible(size, divisor=int(stride))
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size

# endregion


# region Access

def get_channel(
    input   : torch.Tensor | np.ndarray,
    index   : int | tuple[int, int] | list[int],
    keep_dim: bool = True,
) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image.

    Args:
        input   : An image.
        index   : The channel's index.
        keep_dim: If ``True``, keep the dimensions of the return output.
            Default: ``True``.
    """
    if isinstance(index, int):
        i1 = index
        i2 = None if i1 < 0 else i1 + 1
    elif isinstance(index, (list, tuple)):
        i1 = index[0]
        i2 = index[1]
    else:
        raise TypeError
    
    if is_channel_first_image(input=input):
        if input.ndim == 5:
            if keep_dim:
                return input[:, :, i1:i2, :, :] if i2 is not None else input[:, :, i1:, :, :]
            else:
                return input[:, :, i1, :, :] if i2 is not None else input[:, :, i1, :, :]
        elif input.ndim == 4:
            if keep_dim:
                return input[:, i1:i2, :, :] if i2 is not None else input[:, i1:, :, :]
            else:
                return input[:, i1, :, :] if i2 is not None else input[:, i1, :, :]
        elif input.ndim == 3:
            if keep_dim:
                return input[i1:i2, :, :] if i2 is not None else input[i1:, :, :]
            else:
                return input[i1, :, :] if i2 is not None else input[i1, :, :]
        else:
            raise ValueError
    else:
        if input.ndim == 5:
            if keep_dim:
                return input[:, :, :, :, i1:i2] if i2 is not None else input[:, :, :, :, i1:]
            else:
                return input[:, :, :, :, i1] if i2 is not None else input[:, :, :, :, i1]
        elif input.ndim == 4:
            if keep_dim:
                return input[:, :, :, i1:i2] if i2 is not None else input[:, :, :, i1:]
            else:
                return input[:, :, :, i1] if i2 is not None else input[:, :, :, i1]
        elif input.ndim == 3:
            if keep_dim:
                return input[:, :, i1:i2] if i2 is not None else input[:, :, i1:]
            else:
                return input[:, :, i1] if i2 is not None else input[:, :, i1]
        else:
            raise ValueError
    

def get_first_channel(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image."""
    return get_channel(input=input, index=0, keep_dim=True)


def get_last_channel(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image."""
    return get_channel(input=input, index=-1, keep_dim=True)


def get_image_num_channels(input: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        input: An image in channel-last or channel-first format.
    """
    if not 2 <= input.ndim <= 4:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``2`` and ``4``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
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
            f":param:`input`'s number of dimensions must be >= ``3``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
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
            f":param:`input`'s number of dimensions must be >= ``3``, "
            f"but got {input.ndim}."
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
        std  = std.view(-1, 1, 1)  if std.ndim == 1 else std
        input.sub_(mean).div_(std)
    elif isinstance(input, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
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
            f":param:`input`'s number of dimensions must be >= ``3``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
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


def to_3d_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2D or 4D image to a 3D.

    Args:
        input: An image in channel-first format.

    Return:
        A 3D image in channel-first format.
    """
    if not 2 <= input.ndim <= 4:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``2`` and ``4``, "
            f"but got {input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        if input.ndim == 2:  # HW -> 1HW
            input = input.unsqueeze(dim=0)
        elif input.ndim == 4 and input.shape[0] == 1:  # 1CHW -> CHW
            input = input.squeeze(dim=0)
    elif isinstance(input, np.ndarray):
        if input.ndim == 2:  # HW -> 1HW
            input = np.expand_dims(input, axis=0)
        elif input.ndim == 4 and input.shape[0] == 1:  # 1CHW -> CHW
            input = np.squeeze(input, axis=0)
    else:
        raise TypeError(
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
        )
    return input


def to_list_of_3d_image(input: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :class:`list` of 3D images.
   
    Args:
        input: An image of arbitrary type.
        
    Return:
        A :class:`list` of 3D images.
    """
    if isinstance(input, (torch.Tensor, np.ndarray)):
        if input.ndim == 3:
            input = [input]
        elif input.ndim == 4:
            input = list(input)
        else:
            raise ValueError
    elif isinstance(input, list | tuple):
        if not all(isinstance(i, (torch.Tensor, np.ndarray)) for i in input):
            raise ValueError
    return input


def to_4d_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, or 5D image to a 4D.

    Args:
        input: An image in channel-first format.

    Return:
        A 4D image in channel-first format.
    """
    if not 2 <= input.ndim <= 5:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``2`` and ``5``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
        )
    return input


def to_5d_image(input: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 4D, or 6D image to a 5D.
    
    Args:
        input: An tensor in channel-first format.

    Return:
        A 5D image in channel-first format.
    """
    if not 2 <= input.ndim <= 6:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``2`` and ``6``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
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
            f":param:`input`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
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
            f":param:`input`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {input.ndim}."
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
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
            a 3D shape. Default: ``True``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``True``.

    Returns:
        An :class:`numpy.ndarray` image.
    """
    if not 3 <= input.ndim <= 5:
        raise ValueError(
            f":param:`input`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {input.ndim}."
        )
    if isinstance(input, torch.Tensor):
        input = input.detach()
        input = input.cpu().numpy()
    input = denormalize_image(input=input).astype(np.uint8) if denormalize else input
    input = to_channel_last_image(input=input)
    if not keepdim:
        input = to_3d_image(input=input)
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
            to a 4D shape. Default: ``True``.
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
            f":param:`input` must be a :class:`np.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
        )
    input = to_channel_first_image(input=input)
    if not keepdim:
        input = to_4d_image(input=input)
    input = normalize_image(input=input) if normalize else input
    # Place in memory
    input = input.contiguous()
    if device is not None:
        input = input.to(device)
    return input

# endregion


# region I/O

def read_image(
    path     : pathlib.Path,
    to_rgb   : bool = True,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read an image from a file path using :mod:`cv2`. Optionally, convert it
    to RGB format, and :class:`torch.Tensor` type of shape :math:`[1, C, H, W]`.

    Args:
        path: An image file path.
        to_rgb: If ``True``, convert the image from BGR to RGB.
            Default: ``True``.
        to_tensor: If ``True``, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Default: ``False``.
        normalize: If ``True``, normalize the image to :math:`[0.0, 1.0]`.
            Default: ``False``.
        
    Return:
        A :class:`numpy.ndarray` image of shape0 :math:`[H, W, C]` with value in
        range :math:`[0, 255]` or a :class:`torch.Tensor` image of shape
        :math:`[1, C, H, W]` with value in range :math:`[0.0, 1.0]`.
    """
    image = cv2.imread(str(path))  # BGR
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    if to_tensor:
        image = to_image_tensor(input=image, keepdim=False, normalize=normalize)
    return image


def write_image(
    path       : pathlib.Path,
    image      : torch.Tensor | np.ndarray,
    denormalize: bool = False
):
    """Write an image to a file path.
    
    Args:
        image: An image to write.
        path: A directory to write the image to.
        denormalize: If ``True``, convert the image to :math:`[0, 255]`.
            Default: ``False``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def write_image_cv(
    image      : torch.Tensor | np.ndarray,
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = False
):
    """Write an image to a directory using :mod:`cv2`.
    
    Args:
        image: An image to write.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :param:`name`.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to :math:`[0, 255]`.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, torch.Tensor):
        image = to_image_nparray(input=image, keepdim=True, denormalize=denormalize)
    image = to_channel_last_image(input=image)
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"img's number of dimensions must be between ``2`` and ``3``, "
            f"but got {image.ndim}."
        )
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}"  if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    cv2.imwrite(str(file_path), image)


def write_image_torch(
    image      : torch.Tensor | np.ndarray,
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = False
):
    """Write an image to a directory.
    
    Args:
        image: An image to write.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :param:`name`.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to :math:`[0, 255]`.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = to_channel_first_image(input=image)
    image = denormalize_image(image=image) if denormalize else image
    image = image.to(torch.uint8)
    image = image.cpu()
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"img's number of dimensions must be between ``2`` and ``3``, "
            f"but got {image.ndim}."
        )
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    if extension in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(file_path))
    elif extension in [".png"]:
        torchvision.io.image.write_png(input=image, filename=str(file_path))


def write_images_cv(
    images     : list[torch.Tensor | np.ndarray],
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = False
):
    """Write a :class:`list` of images to a directory using :mod:`cv2`.
   
    Args:
        images: A :class:`list` of 3D images.
        dir_path: A directory to write the images to.
        names: A :class:`list` of images' names.
        prefixes: A prefix to add to the :param:`names`.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``False``.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of :param:`images` and :param:`names` must be the same, "
            f"but got {len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of :param:`images` and :param:`prefixes` must be the "
            f"same, but got {len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_cv)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_images_torch(
    images     : Sequence[torch.Tensor | np.ndarray],
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = False
):
    """Write a :class:`list` of images to a directory using :mod:`torchvision`.
   
    Args:
        images: A :class:`list` of 3D images.
        dir_path: A directory to write the images to.
        names: A :class:`list` of images' names.
        prefixes: A prefix to add to the :param:`names`.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`.
            Default: ``False``.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of :param:`images` and :param:`names` must be the same, "
            f"but got {len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of :param:`images` and :param:`prefixes` must be the "
            f"same, but got {len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_torch)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )

# endregion
