#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the basic functionalities of image data."""

from __future__ import annotations

__all__ = [
    "add_weighted",
    "blend",
    "check_image_size",
    "denormalize_image",
    "denormalize_image_mean_std",
    "draw_bbox",
    "draw_heatmap",
    "draw_trajectory",
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
    "make_imgsz_divisible",
    "normalize_image",
    "normalize_image_by_range",
    "normalize_image_mean_std",
    "parse_hw",
    "read_image",
    "read_image_shape",
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
import math
import multiprocessing
from typing import Any, Sequence

import cv2
import joblib
import numpy as np
import torch
import torchvision

from mon.core import error_console, pathlib, utils
from mon.core.typing import _size_any_t


# region Assert

def is_channel_first_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format. We assume
    that if the first dimension is the smallest.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {image.ndim}."
        )
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
    return False


def is_channel_last_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format."""
    return not is_channel_first_image(image=image)


def is_color_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ```True``` if an image is a color image. It is assumed that the
    image has ``3`` or ``4`` channels.
    """
    if get_image_num_channels(image=image) in [3, 4]:
        return True
    return False


def is_gray_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a gray image. It is assumed that the
    image has one channel.
    """
    if get_image_num_channels(image=image) in [1] or len(image.shape) == 2:
        return True
    return False


def is_color_or_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ```True``` if an image is a color or gray image.
    """
    return is_color_image(image) or is_gray_image(image)


def is_image(image: torch.Tensor, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly :math:`[0, 1]` for
    :class:`float` or :math:`[0, 2 ** bits]` for :class:`int`.

    Args:
        image: Image tensor to evaluate.
        bits: The image bits. The default checks if given :class:`int` input
            image is an 8-bit image :math:`[0-255]` or not.

    Raises:
        TypeException: if all the input tensor has not
        1) a shape :math:`[3, H, W]`,
        2) :math:`[0, 1]` for :class:`float` or :math:`[0, 255]` for :class:`int`,
        3) and raises is ``True``.
    
    Example:
        >>> img = torch.rand(2, 3, 4, 4)
        >>> is_image(img)
        True
    """
    if not isinstance(image, torch.Tensor | np.ndarray):
        return False
    res = is_color_or_image(image)
    if not res:
        return False
    '''
    if (
        input.dtype in [torch.float16, torch.float32, torch.float64]
        and (input.min() < 0.0 or input.max() > 1.0)
    ):
        return False
    elif input.min() < 0 or input.max() > 2 ** bits - 1:
        return False
    '''
    return True


def is_integer_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is integer-encoded."""
    c = get_image_num_channels(image=image)
    if c == 1:
        return True
    return False


def is_normalized_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is normalized."""
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )


def is_one_hot_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is one-hot encoded."""
    c = get_image_num_channels(image=image)
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
    new_size = make_imgsz_divisible(size, divisor=int(stride))
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size

# endregion


# region Access

def get_channel(
    image   : torch.Tensor | np.ndarray,
    index   : int | tuple[int, int] | list[int],
    keep_dim: bool = True,
) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image.

    Args:
        image   : An image.
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
    
    if is_channel_first_image(image=image):
        if image.ndim == 5:
            if keep_dim:
                return image[:, :, i1:i2, :, :] if i2 else image[:, :, i1:, :, :]
            else:
                return image[:, :, i1, :, :] if i2 else image[:, :, i1, :, :]
        elif image.ndim == 4:
            if keep_dim:
                return image[:, i1:i2, :, :] if i2 else image[:, i1:, :, :]
            else:
                return image[:, i1, :, :] if i2  else image[:, i1, :, :]
        elif image.ndim == 3:
            if keep_dim:
                return image[i1:i2, :, :] if i2 else image[i1:, :, :]
            else:
                return image[i1, :, :] if i2 else image[i1, :, :]
        else:
            raise ValueError
    else:
        if image.ndim == 5:
            if keep_dim:
                return image[:, :, :, :, i1:i2] if i2 else image[:, :, :, :, i1:]
            else:
                return image[:, :, :, :, i1] if i2 else image[:, :, :, :, i1]
        elif image.ndim == 4:
            if keep_dim:
                return image[:, :, :, i1:i2] if i2 else image[:, :, :, i1:]
            else:
                return image[:, :, :, i1] if i2 else image[:, :, :, i1]
        elif image.ndim == 3:
            if keep_dim:
                return image[:, :, i1:i2] if i2 else image[:, :, i1:]
            else:
                return image[:, :, i1] if i2 else image[:, :, i1]
        else:
            raise ValueError
    

def get_first_channel(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image."""
    return get_channel(image=image, index=0, keep_dim=True)


def get_last_channel(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image."""
    return get_channel(image=image, index=-1, keep_dim=True)


def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An image in channel-last or channel-first format.
    """
    if image.ndim == 4:
        if is_channel_first_image(image=image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
    elif image.ndim == 3:
        if is_channel_first_image(image=image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
    elif image.ndim == 2:
        c = 1
    else:
        # error_console.log(
        #     f":param:`image`'s number of dimensions must be between ``2`` and ``4``, "
        #     f"but got {input.ndim}."
        # )
        c = 0
    return c


def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as :math:`(x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image=image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    :math:`(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image=image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )


def get_image_size(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height and width value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first_image(image=image):
        return [image.shape[-2], image.shape[-1]]
    else:
        return [image.shape[-3], image.shape[-2]]


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first_image(image=image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]

# endregion


# region Convert

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
        raise ValueError(
            f":param:`image`'s number of dimensions must be >= ``3``, "
            f"but got {image.ndim}."
        )
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
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1) if std_inv.ndim == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
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
        raise ValueError(
            f":param:`image`'s number of dimensions must be >= ``3``, "
            f"but got {image.ndim}."
        )
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
        std  = std.view(-1, 1, 1)  if std.ndim == 1 else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range [:param:`min`, :param:`max`] to the
    [:param:`new_min`, :param:`new_max`].
    
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
        raise ValueError(
            f":param:`image`'s number of dimensions must be >= ``3``, "
            f"but got {image.ndim}."
        )
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
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
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


def to_3d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D or 4D image to a 3D.

    Args:
        image: An image in channel-first format.

    Return:
        A 3D image in channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between ``2`` and ``4``, "
            f"but got {image.ndim}."
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
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def to_list_of_3d_image(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :class:`list` of 3D images.
   
    Args:
        image: An image of arbitrary type.
        
    Return:
        A :class:`list` of 3D images.
    """
    if isinstance(image, torch.Tensor | np.ndarray):
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


def to_4d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 5D, list of 3D, and list of 4D images to 4D.

    Args:
        image: A 2D, 3D, 5D, list of 3D, and list of 4D images in channel-first format.

    Return:
        A 4D image in channel-first format.
    """
    if isinstance(image, torch.Tensor | np.ndarray) and not 2 <= image.ndim <= 5:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between "
            f"``2`` and ``5``, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 11HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1NCHW -> NCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 11HW
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1NCHW -> NHWC
            image = np.squeeze(image, axis=0)
    elif isinstance(image, list | tuple):
        if all(isinstance(i, torch.Tensor)   and i.ndim == 3 for i in image):
            image = torch.stack(image, dim=0)
        elif all(isinstance(i, torch.Tensor) and i.ndim == 4 for i in image):
            image = torch.cat(image, dim=0)
        elif all(isinstance(i, np.ndarray)   and i.ndim == 3 for i in image):
            image = np.array(image)
        elif all(isinstance(i, np.ndarray)   and i.ndim == 4 for i in image):
            image = np.concatenate(image, axis=0)
        # else:
        #     error_console.log(f"input's number of dimensions must be between ``3`` and ``4``.")
        #     image = None
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray`, :class:`torch.Tensor`, "
            f"or a :class:`list` of either of them, but got {type(image)}."
        )
    return image


def to_5d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 4D, or 6D image to a 5D.
    
    Args:
        image: An tensor in channel-first format.

    Return:
        A 5D image in channel-first format.
    """
    if not 2 <= image.ndim <= 6:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between ``2`` and ``6``, "
            f"but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 111HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 11CHW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4:  # NCHW -> 1NCHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 6 and image.shape[0] == 1:  # 1*NCHW -> *NCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 111HW
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # HWC -> 11HWC
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 4:  # BHWC -> 1BHWC
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 6 and image.shape[0] == 1:  # 1*BHWC -> *BHWC
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def to_channel_first_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format.
    
    Args:
        image: An image in channel-last or channel-first format.
    
    Returns:
        An image in channel-first format.
    """
    if is_channel_first_image(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 5:
            image = image.permute(0, 1, 4, 2, 3)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 4, 2, 3))
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def to_channel_last_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format.

    Args:
        image: An image in channel-last or channel-first format.

    Returns:
        A image in channel-last format.
    """
    if is_channel_last_image(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between ``3`` and ``5``, "
            f"but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)
        elif image.ndim == 5:
            image = image.permute(0, 1, 3, 4, 2)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 3, 4, 2))
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def to_image_nparray(
    image      : torch.Tensor | np.ndarray,
    keepdim    : bool = False,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :class:`numpy.ndarray`.
    
    Args:
        image: An image.
        keepdim: If `True`, keep the original shape. If ``False``, convert it to
            a 3D shape. Default: ``True``.
        denormalize: If ``True``, convert image to :math:`[0, 255]`. Default: ``True``.

    Returns:
        An :class:`numpy.ndarray` image.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f":param:`image`'s number of dimensions must be between "
            f"``3`` and ``5``, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.detach()
        image = image.cpu().numpy()
    image = denormalize_image(image=image).astype(np.uint8) if denormalize else image
    image = to_channel_last_image(image=image)
    if not keepdim:
        image = to_3d_image(image=image)
    return image


def to_image_tensor(
    image    : torch.Tensor | np.ndarray,
    keepdim  : bool = False,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :class:`PIL.Image` or :class:`numpy.ndarray` to
    :class:`torch.Tensor`. Optionally, convert :param:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An image in channel-last or channel-first format.
        keepdim: If ``True``, keep the original shape. If ``False``, convert it
            to a 4D shape. Default: ``True``.
        normalize: If ``True``, normalize the image to :math:``[0, 1]``.
            Default: ``False``.
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        
    Returns:
        A :class:`torch.Tensor` image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(image)}."
        )
    image = to_channel_first_image(image=image)
    if not keepdim:
        image = to_4d_image(image=image)
    image = normalize_image(image=image) if normalize else image
    # Place in memory
    image = image.contiguous()
    if device:
        image = image.to(device)
    return image

# endregion


# region Draw

def draw_bbox(
    image     : np.ndarray,
    bbox      : np.ndarray | list,
    label     : int | str | None = None,
    color     : list[int]        = [255, 255, 255],
    thickness : int              = 1,
    line_type : int              = cv2.LINE_8,
    shift     : int              = 0,
    font_face : int              = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: int              = 0.8,
    fill      : bool | float     = False,
) -> np.ndarray:
    """Draw a bounding box on an image.
    
    Args:
        image: An image.
        bbox: A bounding box in XYXY format.
        label: A label for the bounding box.
        color: A color of the bounding box.
        thickness: The thickness of the rectangle borderline in px. Thickness
            of ``-1 px`` will fill the rectangle shape by the specified color.
            Default: ``1``.
        line_type: The type of the line. One of:
            - ``'cv2.LINE_4'``  - 4-connected line.
            - ``'cv2.LINE_8'``  - 8-connected line (default).
            - ``'cv2.LINE_AA'`` - antialiased line.
            Default: ``'cv2.LINE_8'``.
        font_face: The font of the label's text. Default: ``cv2.FONT_HERSHEY_DUPLEX``.
        font_scale: The scale of the label's text. Default: ``0.8``.
        shift: The number of fractional bits in the point coordinates.
            Default: ``0``.
        fill: Fill the region inside the bounding box with transparent color.
            A float value :math:``[0.0-1.0]`` indicates the transparency ratio.
            A ``True`` value means ``0.5``. A value of ``1.0`` equals to
            :param:`thickness` = -1. Default: ``False``.
    """
    drawing = image.copy()
    color   = color or [255, 255, 255]
    pt1     = (int(bbox[0]), int(bbox[1]))
    pt2     = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(
        img       = drawing,
        pt1       = pt1,
        pt2       = pt2,
        color     = color,
        thickness = thickness,
        lineType  = line_type,
        shift     = shift,
    )
    if label not in [None, "None", ""]:
        label  = f"{label}"
        offset = int(thickness / 2)
        text_size, baseline = cv2.getTextSize(
            text      = label,
            fontFace  = font_face,
            fontScale = font_scale,
            thickness = 1
        )
        cv2.rectangle(
            img       = image,
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED,
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(
            img       = image,
            text      = label,
            org       = text_org,
            fontFace  = font_face,
            fontScale = font_scale,
            color     = [255, 255, 255],
            thickness = 1
        )
    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = image.copy()
        cv2.rectangle(
            img       = overlay,
            pt1       = pt1,
            pt2       = pt2,
            color     = color,
            thickness = -1,
        )
        cv2.addWeighted(
            src1  = overlay,
            alpha = alpha,
            src2  = drawing,
            beta  = 1 - alpha,
            gamma = 0,
            dst   = drawing,
        )
    return drawing


def draw_heatmap(
    image     : np.ndarray,
    mask      : np.ndarray,
    color_map : int   = cv2.COLORMAP_JET,
    alpha     : float = 0.5,
    use_rgb   : bool  = False,
) -> np.ndarray:
    """Overlay a mask on the image as a heatmap. By default, the heatmap is in
    BGR format.
    
    Args:
        image: An image in RGB or BGR format.
        mask: A heatmap mask.
        color_map: A color map for the heatmap. Default: ``cv2.COLORMAP_JET``.
        alpha: The transparency ratio of the image. The final result is:
            :math:`alpha * image + (1 - alpha) * mask`. Default: ``0.5``.
        use_rgb: If ``True``, convert the heatmap to RGB format. Default: ``False``.
    
    Returns:
        An image with the heatmap overlay.
    """
    if np.max(image) > 1:
        raise Exception(":param:`image` should be an :class:`np.float32` in the range :math:`[0, 1]`.")
    if not 0 <= alpha <= 1:
        raise Exception(f":param:`alpha` should be in the range :math:`[0, 1]`, but got: {alpha}.")
    
    if is_normalized_image(mask):
        mask = np.uint8(255 * mask)
    heatmap  = cv2.applyColorMap(np.uint8(255 * mask), color_map)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap  = np.float32(heatmap) / 255
    
    drawing = (1 - alpha) * heatmap + alpha * image
    drawing = drawing / np.max(drawing)
    drawing = np.uint8(255 * drawing)
    return drawing


def draw_trajectory(
    image     : np.ndarray,
    trajectory: np.ndarray | list,
    color     : list[int] = [255, 255, 255],
    thickness : int       = 1,
    line_type : int       = cv2.LINE_8,
    point     : bool      = False,
    radius    : int       = 3,
) -> np.ndarray:
    """Draw a trajectory path on an image.
    
    Args:
        image: An image.
        trajectory: A 2D array or list of points in :math:`[(x1, y1), ...]` format.
        color: A color of the bounding box.
        thickness: The thickness of the path in px. Default: 1.
        line_type: The type of the line. One of:
            - ``'cv2.LINE_4'``  - 4-connected line.
            - ``'cv2.LINE_8'``  - 8-connected line (default).
            - ``'cv2.LINE_AA'`` - antialiased line.
            Default:``' cv2.LINE_8'``.
        point: If ``True``, draw each point along the trajectory path. Default: ``False``.
        radius: The radius value of the point. Default: ``3``.
    """
    drawing = image.copy()
    
    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError(f":param:`trajectory` must be a list of points in [(x1, y1), ...] format.")
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(
        img       = drawing,
        pts       = [trajectory],
        isClosed  = False,
        color     = color,
        thickness = thickness,
        lineType  = line_type,
    )
    if point:
        for p in trajectory:
            cv2.circle(
                img       = drawing,
                center    = p[0],
                radius    = radius,
                thickness = -1,
                color     = color
            )
    return drawing

# endregion


# region I/O

def read_image(
    path     : pathlib.Path,
    flags    : int  = cv2.IMREAD_COLOR,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read an image from a file path using :mod:`cv2`. Optionally, convert it
    to RGB format, and :class:`torch.Tensor` type of shape :math:`[1, C, H, W]`.

    Args:
        path: An image's file path.
        flags: A flag to read the image. One of:
            - cv2.IMREAD_UNCHANGED           = -1,
            - cv2.IMREAD_GRAYSCALE           = 0,
            - cv2.IMREAD_COLOR               = 1,
            - cv2.IMREAD_ANYDEPTH            = 2,
            - cv2.IMREAD_ANYCOLOR            = 4,
            - cv2.IMREAD_LOAD_GDAL           = 8,
            - cv2.IMREAD_REDUCED_GRAYSCALE_2 = 16,
            - cv2.IMREAD_REDUCED_COLOR_2     = 17,
            - cv2.IMREAD_REDUCED_GRAYSCALE_4 = 32,
            - cv2.IMREAD_REDUCED_COLOR_4     = 33,
            - cv2.IMREAD_REDUCED_GRAYSCALE_8 = 64,
            - cv2.IMREAD_REDUCED_COLOR_8     = 65,
            - cv2.IMREAD_IGNORE_ORIENTATION  = 128
            Default: ``cv2.IMREAD_COLOR``.
        to_tensor: If ``True``, convert the image from :class:`numpy.ndarray`
            to :class:`torch.Tensor`. Default: ``False``.
        normalize: If ``True``, normalize the image to :math:`[0.0, 1.0]`. Default: ``False``.
        
    Return:
        A :class:`numpy.ndarray` image of shape0 :math:`[H, W, C]` with value in
        range :math:`[0, 255]` or a :class:`torch.Tensor` image of shape
        :math:`[1, C, H, W]` with value in range :math:`[0.0, 1.0]`.
    """
    image = cv2.imread(str(path), flags)  # BGR
    if image.ndim == 2:  # HW -> HW1 (OpenCV read grayscale image)
        image = np.expand_dims(image, axis=-1)
    if is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if to_tensor:
        image = to_image_tensor(image=image, keepdim=False, normalize=normalize)
    return image


def read_image_shape(path: pathlib.Path) -> tuple[int, ...]:
    """Read an image from a file path using :mod:`cv2` and get its shape as
    :math:`[H, W, C]`.
    
    Args:
        path: An image file path.
    """
    image = cv2.imread(str(path))  # BGR
    return image.shape
    

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
        image = to_image_nparray(image=image, keepdim=True, denormalize=denormalize)
    image = to_channel_last_image(image=image)
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"image's number of dimensions must be between ``2`` and ``3``, "
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
        image = to_channel_first_image(image=image)
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


# region Ops

def add_weighted(
    image1: torch.Tensor | np.ndarray,
    alpha : float,
    image2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image.
        alpha: The weight of the :param:`image1` elements.
        image2: The second image.
        beta: The weight of the :param:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"The shape of :param:`image1` and :param:`image2` must be the same, "
            f"but got {image1.shape} and {image2.shape}."
        )
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound)
    else:
        raise TypeError(
            f":param:`image` must be a :class:`numpy.ndarray` or :class:`torch.Tensor`, "
            f"but got {type(input)}."
        )
    return output


def blend(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :param:`image1` * alpha + :param:`image2` * beta + gamma

    Args:
        image1: A source image.
        image2: A n overlay image that we want to blend on top of :param:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion


# region Parsing

def make_imgsz_divisible(input: Any, divisor: int = 32) -> int | tuple[int, int]:
    """Make an image sizes divisible by a given stride.
    
    Args:
        input: An image size, size, or shape.
        divisor: The divisor. Default: ``32``.
    
    Returns:
        A new image size.
    """
    h, w = parse_hw(input)
    h    = int(math.ceil(h / divisor) * divisor)
    w    = int(math.ceil(w / divisor) * divisor)
    return h, w


def parse_hw(size: _size_any_t) -> list[int]:
    """Casts a size object to the standard :math:`[H, W]`.

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
