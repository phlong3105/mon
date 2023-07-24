#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for images. They're the most basic
operations performed on an image like getting metadata or properties, basic type
conversion.
"""

from __future__ import annotations

__all__ = [
    "add_weighted", "blend", "check_image_size", "denormalize_image",
    "denormalize_image_mean_std", "get_image_center", "get_image_center4",
    "get_image_shape", "get_image_size", "get_num_channels", "is_channel_first",
    "is_channel_last", "is_color_image", "is_integer_image",
    "is_normalized_image", "is_one_hot_image", "normalize_image",
    "normalize_image_by_range", "normalize_image_mean_std", "to_3d", "to_4d",
    "to_5d", "to_channel_first", "to_channel_last", "to_list_of_3d",
    "to_nparray", "get_hw", "to_tensor", "upcast",
]

import copy
import functools
from typing import Any

import multipledispatch
import numpy as np
import torch

from mon.coreml import device as md
from mon.foundation import error_console, math


# region Obtainment

def is_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Return True if an image is in the channel-first format. It is assumed
    that if the first dimension is the smallest.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
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


def is_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Return True if an image is in the channel-first format."""
    return not is_channel_first(image=image)


def is_color_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return True if an image is a color image. It is assumed that the image
    has 3 or 4 channels.
    """
    if get_num_channels(image=image) in [3, 4]:
        return True
    return False


def is_integer_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return True ian image is integer-encoded."""
    c = get_num_channels(image=image)
    if c == 1:
        return True
    return False


def is_normalized_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return True if an image is normalized."""
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def is_one_hot_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return True if an image is one-hot encoded."""
    c = get_num_channels(image=image)
    if c > 1:
        return True
    return False


def check_image_size(size: list[int], stride: int = 32) -> int:
    """If the input :param:`size` isn't a multiple of the :param:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: An image's size.
        stride: The stride of a network. Defaults to 32.
    
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


def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as (x=h/2, y=w/2).
    
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
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as (x=h/2, y=w/2, x=h/2,
    y=w/2).
    
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
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )


def get_image_size(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height and width value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first(image=image):
        return [image.shape[-2], image.shape[-1]]
    else:
        return [image.shape[-3], image.shape[-2]]


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An image
    """
    if is_channel_first(image=image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]


def get_hw(size: int | list[int]) -> list[int]:
    """Casts a size object to the standard HW.

    Args:
        size: A size of an image, windows, or kernels, etc.

    Returns:
        A size in HW format.
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


def get_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An image in channel-last or channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(
            f"image's number of dimensions must be between 2 and 4, but got "
            f"{image.ndim}."
        )
    if image.ndim == 4:
        if is_channel_first(image=image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
    elif image.ndim == 3:
        if is_channel_first(image=image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
    else:
        c = 1
    return c

# endregion


# region Creation

@multipledispatch.dispatch(int, torch.Tensor)
def eye_like(n: int, x: torch.Tensor) -> torch.Tensor:
    """Create a tensor of shape `(n, n)` with ones on the diagonal and zeros
    everywhere else, and then repeats it along the batch dimension to match the
    shape of the input tensor.

    Args:
        n: The number of rows and columns in the output tensor.
        x: An input tensor.

    Return:
        A tensor of shape (input.shape[0], n, n).
    """
    if not x.ndim >= 1:
        raise ValueError(
            f"x's number of dimensions must be >= 1, but got {x.ndim}."
        )
    if not n > 0:
        raise ValueError(f"n must be > 0, but got {n}.")
    identity = torch.eye(n, device=x.device, dtype=x.dtype)
    return identity[None].repeat(x.shape[0], 1, 1)


@multipledispatch.dispatch(int, torch.Tensor)
def vec_like(n: int, x: torch.Tensor) -> torch.Tensor:
    """Create a vector of zeros with the same shape as the input.

    Args:
        n: The number of elements in the vector.
        x: An input tensor.

    Return:
        A tensor of zeros with the same shape as the input tensor.
    """
    if not x.ndim >= 1:
        raise ValueError(
            f"x's number of dimensions must be >= 1, but got {x.ndim}."
        )
    if not n > 0:
        raise ValueError(f"n must be > 0, but got {n}.")
    vec = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
    return vec[None].repeat(x.shape[0], 1, 1)

# endregion


# region Alteration

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
        mean: A sequence of means for each channel. Defaults to
            [0.485, 0.456, 0.406].
        std: A sequence of standard deviations for each channel. Defaults to
            [0.229, 0.224, 0.225].
        eps: A scalar value to avoid zero divisions. Defaults to 1e-6.
        
    Returns:
        A denormalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.devices
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.devices)
        
        std_inv  = 1.0 / (std + eps)
        mean_inv = -mean * std_inv
        std_inv  = std_inv.view(-1, 1, 1) if std_inv.ndim == 1 else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
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
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image in channel-first format.
        mean: A sequence of means for each channel. Defaults to
            [0.485, 0.456, 0.406].
        std: A sequence of standard deviations for each channel. Defaults to
            [0.229, 0.224, 0.225].
        eps: A scalar value to avoid zero divisions. Defaults to 1e-6.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        shape  = image.shape
        device = image.devices
        dtype  = image.dtype
        if isinstance(mean, float):
            mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
        elif isinstance(mean, (list, tuple)):
            mean = torch.as_tensor(mean, dtype=dtype, device=image.devices)
        elif isinstance(mean, torch.Tensor):
            mean = mean.to(dtype=dtype, device=image.devices)
        
        if isinstance(std, float):
            std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
        elif isinstance(std, (list, tuple)):
            std = torch.as_tensor(std, dtype=dtype, device=image.devices)
        elif isinstance(std, torch.Tensor):
            std = std.to(dtype=dtype, device=image.devices)
        std += eps
        
        mean = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std  = std.view(-1, 1, 1) if std.ndim == 1 else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def normalize_image_by_range(
    image  : torch.Tensor | np.ndarray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> torch.Tensor | np.ndarray:
    """Normalize an image from the range [:param:`min`, :param:`max`] to
    the [:param:`new_min`, :param:`new_max`].
    
    Args:
        image: An image.
        min: The current minimum pixel value of the image. Defaults to 0.0.
        max: The current maximum pixel value of the image. Defaults to 255.0.
        new_min: A new minimum pixel value of the image. Defaults to 0.0.
        new_max: A new minimum pixel value of the image. Defaults to 1.0.
        
    Returns:
        A normalized image.
    """
    if not image.ndim >= 3:
        raise ValueError(
            f"image's number of dimensions must be >= 3, but got {image.ndim}."
        )
    # if is_normalized_image(image=image):
    #     return image
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(image, np.ndarray):
        image = copy.deepcopy(image)
        image = image.astype(np.float32)
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = np.cip(image, new_min, new_max)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
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


def to_3d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
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


def to_list_of_3d(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a list of 3-D images.
   
    Args:
        image: An image of arbitrary type.
        
    Return:
        A list of 3-D images.
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
    

def to_4d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, or 5-D image to a 4-D.

    Args:
        image: An image in channel-first format.

    Return:
        A 4-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 5:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 5, but got "
            f"{image.ndim}."
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
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_5d(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a 2-D, 3-D, 4-D, or 6-D image to a 5-D.
    
    Args:
        image: An tensor in channel-first format.

    Return:
        A 5-D image in channel-first format.
    """
    if not 2 <= image.ndim <= 6:
        raise ValueError(
            f"x's number of dimensions must be between 2 and 6, but got "
            f"{image.ndim}."
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
            f"x must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


def to_channel_first(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format.
    
    Args:
        image: An image in channel-last or channel-first format.
    
    Returns:
        An image in channel-first format.
    """
    if is_channel_first(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
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
            f"image must be torch.Tensor or a numpy.ndarray, but got {type(image)}."
        )
    return image


def to_channel_last(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format.

    Args:
        image: An image in channel-last or channel-first format.

    Returns:
        A image in channel-last format.
    """
    if is_channel_last(image=image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
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
            f"image must be torch.Tensor or a numpy.ndarray, but got {type(image)}."
        )
    return image


def to_nparray(
    image      : torch.Tensor | np.ndarray,
    keepdim    : bool = True,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :class:`numpy.ndarray`.
    
    Args:
        image: An image.
        keepdim: If True, keep the original shape. If False, convert it to a 3-D
            shape. Defaults to True.
        denormalize: If True, convert image to [0, 255]. Defaults to True.

    Returns:
        An :class:`numpy.ndarray` image.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(
            f"image's number of dimensions must be between 3 and 5, but got "
            f"{image.ndim}."
        )
    if isinstance(image, torch.Tensor):
        image = image.detach()
        image = image.cpu().numpy()
    image = denormalize_image(image=image).astype(np.uint) if denormalize else image
    image = to_channel_last(image=image)
    if not keepdim:
        image = to_3d(image=image)
    return image


def to_tensor(
    image    : torch.Tensor | np.ndarray,
    keepdim  : bool = True,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :class:`PIL.Image` or :class:`numpy.ndarray` to
    :class:`torch.Tensor`. Optionally, convert :param:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An image in channel-last or channel-first format.
        keepdim: If True, keep the original shape. If False, convert it to a 4-D
            shape. Defaults to True.
        normalize: If True, normalize the image to [0, 1]. Defaults to False.
        device: The device to run the model on. If None, the default 'cpu'
            device is used.
        
    Returns:
        A :class:`torch.Tensor` image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    image = to_channel_first(image=image)
    if not keepdim:
        image = to_4d(image=image)
    image = normalize_image(image=image) if normalize else image
    # Place in memory
    image = image.contiguous()
    if device is not None:
        device = md.select_device(device=device) \
            if not isinstance(device, torch.device) else device
        image = image.to(device)
    return image


def upcast(
    image    : torch.Tensor | np.ndarray,
    keep_type: bool = False
) -> torch.Tensor | np.ndarray:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.

    Args:
        image: An image of arbitrary type.
        keep_type: If True, keep the same type (int32 -> int64). Else upcast to
            a higher type (int32 -> float32).
    Return:
        An image of higher type.
    """
    if image.dtype is torch.float16:
        return image.to(torch.float32)
    elif image.dtype is torch.float32:
        return image  # x.to(torch.float64)
    elif image.dtype is torch.int8:
        return image.to(torch.int16) if keep_type else image.to(torch.float16)
    elif image.dtype is torch.int16:
        return image.to(torch.int32) if keep_type else image.to(torch.float32)
    elif image.dtype is torch.int32:
        return image  # x.to(torch.int64) if keep_type else x.to(torch.float64)
    elif type(image) is np.float16:
        return image.astype(np.float32)
    elif type(image) is np.float32:
        return image  # x.astype(np.float64)
    elif type(image) is np.int16:
        return image.astype(np.int32) if keep_type else image.astype(np.float32)
    elif type(image) is np.int32:
        return image  # x.astype(np.int64) if keep_type else x.astype(np.int64)
    return image

# endregion


# region Operation

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
        gamma: A scalar added to each sum. Defaults to 0.0.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"The shape of x and y must be the same, but got {image1.shape} and "
            f"{image2.shape}."
        )
    bound = 1.0 if image1.is_floating_point() else 255.0
    image = image1 * alpha + image2 * beta + gamma
    if isinstance(image, torch.Tensor):
        image = image.clamp(0, bound).to(image1.dtype)
    elif isinstance(image, np.ndarray):
        image = np.clip(image, 0, bound)
    else:
        raise TypeError(
            f"image must be a np.ndarray or torch.Tensor, but got {type(image)}."
        )
    return image


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
        gamma: A scalar added to each sum. Defaults to 0.0.

    Returns:
        Blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion
