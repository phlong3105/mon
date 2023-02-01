#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for images. They're the most basic
operations performed on an image like getting metadata or properties, basic type
conversion.
"""

from __future__ import annotations

__all__ = [
    "add_weighted", "check_image_size", "correct_image_dimension",
    "denormalize_image", "denormalize_mean_std", "get_image_center",
    "get_image_center4", "get_image_hw", "get_image_shape", "get_image_size",
    "get_num_channels", "is_channel_first", "is_channel_last", "is_color_image",
    "is_integer_image", "is_normalized", "is_one_hot_image",
    "normalize_by_range", "normalize_image", "normalize_mean_std",
    "to_channel_first", "to_channel_last", "to_image", "to_pil_image",
    "to_size", "to_tensor",
]

import copy
import functools
from typing import TypeAlias

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import functional

from mon import core
from mon.coreimage.typing import FloatAnyT, Int2T, Int3T, Ints, TensorOrArray
from mon.coreimage.util import tensor


# region Conversion

def correct_image_dimension(image: np.ndarray) -> np.ndarray:
    """Correct the dimensionality of an image to the correct format used by
    :mod:`matplotlib` and :mod:`cv2`.
    
    Args:
        image: An image to be corrected.
    
    Returns:
        A corrected image ready to be used.
    """
    assert isinstance(image, np.ndarray)
    assert 3 <= image.ndim <= 4
    assert is_channel_last(image)
    
    image = copy.deepcopy(image)
    if image.ndim == 2:
        pass
    elif image.ndim == 3:
        if image.shape[-1] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = np.squeeze(image, axis=-1)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        image = np.transpose(image, (0, 2, 3, 1))
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
    return image


def correct_tensor_dimension(image: torch.Tensor) -> torch.Tensor:
    """Correct the dimensionality of image to the correct format used by
    :mod:`torch`. It should be a :class:`torch.Tensor` in channel-first format
    and has shape [..., 1 or B, C, H, W]

    Args:
        image: An image to be corrected.

    Returns:
        A corrected image ready to be used.
    """
    assert isinstance(image, torch.Tensor)
    assert 3 <= image.ndim <= 5
    assert is_channel_first(image)
    
    image = image.clone()
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        pass
    return image


def to_channel_first(image: TensorOrArray) -> TensorOrArray:
    """Convert an image to the channel-first format.
    
    Args:
        image: An image to be converted.
    
    Returns:
        A channel-first image.
    """
    if is_channel_first(image):
        return image

    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 5
    
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
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def to_channel_last(image: TensorOrArray) -> TensorOrArray:
    """Convert an image to the channel-last format.

    Args:
        image: An image to be converted.

    Returns:
        A channel-last image.
    """
    if is_channel_last(image):
        return image

    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 5
    
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
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def to_image(
    image      : torch.Tensor,
    keepdim    : bool = True,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image from :class:`torch.Tensor` to :class:`numpy.ndarray`.
    
    Args:
        image: An image of shape [..., C, H, W] to be converted.
        keepdim: If True, keep the dimensions of the input tensor. Defaults to
            True.
        denormalize: If True, denormalize the image to [0, 255]. Defaults to
            False.
        
    Returns:
        An :class:`numpy.ndarray` image.
    """
    assert isinstance(image, torch.Tensor)
    assert 3 <= image.ndim <= 4
    image = image.clone()
    image = image.detach()
    image = tensor.to_3d_tensor(image)
    image = denormalize_image(image=image) if denormalize else image
    image = to_channel_last(image=image)
    if not keepdim:
        image = correct_image_dimension(image=image)
    image = image.cpu().numpy()
    image = image.astype(np.uint8)
    return image


def to_pil_image(image: TensorOrArray) -> PIL.Image:
    """Convert an image from :class:`torch.Tensor` or :class:`numpy.ndarray` to
    :class:`PIL.Image`.
    
    Args:
        image: An image to be converted.
    
    Returns:
        A :class:`PIL.Image` image.
    """
    if isinstance(image, torch.Tensor):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return functional.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def to_tensor(
    image    : TensorOrArray | PIL.Image,
    keepdim  : bool = True,
    normalize: bool = False,
) -> torch.Tensor:
    """Convert an image from :class:`PIL.Image` or :class:`numpy.ndarray` to
    :class:`torch.Tensor`. Optionally, convert :param:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An image to be converted.
        keepdim: If True, keep the channel dimension. If False unsqueeze the
            image to match the shape [..., C, H, W]. Defaults to True
        normalize: If True, normalize the image to [0, 1]. Defaults to False
    
    Returns:
        A :class:`torch.Tensor` image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray) \
           or functional._is_pil_image(image)
    if isinstance(image, torch.Tensor | np.ndarray):
        assert 2 <= image.ndim <= 4
    
    # Handle :class:`PIL.Image`
    if functional._is_pil_image(image):
        image          = copy.deepcopy(image)
        mode           = image.mode
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        image          = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        if mode == "1":
            image = 255 * image
    
    # Handle :class:`numpy.ndarray`
    if functional._is_numpy(image):
        image = image.copy()
        image = torch.from_numpy(image).contiguous()
    
    # Channel first format
    image = to_channel_first(image=image)
    if not keepdim:
        image = correct_tensor_dimension(image=image)
    
    # Normalize
    if normalize:
        image = normalize_image(image=image)
    
    # Convert type
    if isinstance(image, torch.ByteTensor):
        return image.to(dtype=torch.get_default_dtype())
    
    # Place in memory
    image = image.contiguous()
    return image

# endregion


# region Image Property

def check_image_size(size: Ints, stride: int = 32) -> int:
    """If the input :param:`size` isn't a multiple of the :param:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: A size of the image.
        stride: A stride of the network. Defaults to 32.
    
    Returns:
        A new size of the image.
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:    # [C, H, W]
            size = size[1]
        elif len(size) == 2:  # [H, W]
            size = size[0]
        
    new_size = core.math.make_divisible(size, int(stride))  # ceil gs-multiple
    if new_size != size:
        core.error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


def get_image_center(image: TensorOrArray) -> TensorOrArray:
    """Return the center of a given image specified as (x=h/2, y=w/2).
    
    Args:
        image: An image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    h, w = get_image_hw(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def get_image_center4(image: TensorOrArray) -> TensorOrArray:
    """Return the center of a given image specified as (x=h/2, y=w/2, x=h/2,
    y=w/2).
    
    Args:
        image: An image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    h, w = get_image_hw(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def get_image_hw(image: TensorOrArray) -> Int2T :
    """Return height and width value of an image.
    
    Args:
        image: An image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    # [.., C, H, W]
    if is_channel_first(image):
        return image.shape[-2], image.shape[-1]
    # [.., H, W, C]
    else:
        return image.shape[-3], image.shape[-2]
    

def get_image_shape(image: TensorOrArray) -> Int3T :
    """Return channel, height, and width value of an image.
    
    Args:
        image: An image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    # [.., C, H, W]
    if is_channel_first(image=image):
        return image.shape[-3], image.shape[-2], image.shape[-1]
    # [.., H, W, C]
    else:
        return image.shape[-1], image.shape[-3], image.shape[-2]


get_image_size: TypeAlias = get_image_hw


def get_num_channels(image: TensorOrArray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 4
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
    return 0


def to_size(size: Ints) -> tuple[int, int]:
    """Casts a size object to the standard [H, W].

    Args:
        size: A size of the image, windows, kernels, etc.

    Returns:
        A size in [H, W] format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            size = size[1:3]
        if len(size) == 1:
            size = (size[0], size[0])
    elif isinstance(size, int):
        size = (size, size)
    return tuple(size)

# endregion


# region Image Format (Value Type and Channel Type)

def is_channel_first(image: TensorOrArray) -> bool:
    """Return True if an image is in the channel-first format. It is assumed
    that if the first dimension is the smallest.
    
    Args:
        image: An image to be checked.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 5
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


def is_channel_last(image: TensorOrArray) -> bool:
    """Return True if an image is in the channel-first format.
    
    Args:
        image: An image to be checked.
    """
    return not is_channel_first(image=image)


def is_color_image(image: TensorOrArray) -> bool:
    """Return True if an image is a color image. It is assumed that the image
    has 3 or 4 channels.
    """
    if get_num_channels(image) in [3, 4]:
        return True
    return False


def is_integer_image(image: TensorOrArray) -> bool:
    """Return True ian image is integer-encoded."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    c = get_num_channels(image=image)
    if c == 1:
        return True
    return False


def is_normalized(image: TensorOrArray) -> bool:
    """Return True if an image is normalized."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def is_one_hot_image(image: TensorOrArray) -> bool:
    """Return True if an image is one-hot encoded."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


def denormalize_mean_std(
    image: TensorOrArray,
    mean : FloatAnyT = (0.485, 0.456, 0.406),
    std  : FloatAnyT = (0.229, 0.224, 0.225),
    eps  : float     = 1e-6,
) -> TensorOrArray:
    """Denormalize an image with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image of shape [..., C, H, W] to be adjusted.
        mean: A sequence of means for each channel. Defaults to
            (0.485, 0.456, 0.406).
        std: A sequence of standard deviations for each channel. Defaults to
            (0.229, 0.224, 0.225).
        eps: A scalar value to avoid zero divisions. Defaults to 1e-6.
        
    Returns:
        A denormalized image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    
    if isinstance(image, torch.Tensor):
        image  = image.clone()
        image  = image.to(dtype=torch.get_default_dtype()) \
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
        std_inv  = std_inv.view(-1, 1, 1)  if std_inv.ndim == 1  else std_inv
        mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
        image.sub_(mean_inv).div_(std_inv)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def normalize_mean_std(
    image: TensorOrArray,
    mean : FloatAnyT = (0.485, 0.456, 0.406),
    std  : FloatAnyT = (0.229, 0.224, 0.225),
    eps  : float     = 1e-6,
) -> TensorOrArray:
    """Normalize :param:`image` with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: An image of shape [..., C, H, W] to be adjusted.
        mean: A sequence of means for each channel. Defaults to
            (0.485, 0.456, 0.406).
        std: A sequence of standard deviations for each channel. Defaults to
            (0.229, 0.224, 0.225).
        eps: A scalar value to avoid zero divisions. Defaults to 1e-6.
        
    Returns:
        A normalized image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    
    if isinstance(image, torch.Tensor):
        image  = image.clone()
        image  = image.to(dtype=torch.get_default_dtype()) \
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
        std  += eps
        
        mean  = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
        std   = std.view(-1, 1, 1)  if std.ndim == 1  else std
        image.sub_(mean).div_(std)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def normalize_by_range(
    image  : TensorOrArray,
    min    : float = 0.0,
    max    : float = 255.0,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> TensorOrArray:
    """Normalize an image from the range [:param:`min`, :param:`max`] to
    the [:param:`new_min`, :param:`new_max`].
    
    Args:
        image: An image of shape [..., C, H, W] to be adjusted.
        min: The current minimum pixel value of the image. Defaults to 0.0.
        max: The current maximum pixel value of the image. Defaults to 255.0.
        new_min: A new minimum pixel value of the image. Defaults to 0.0.
        new_max: A new minimum pixel value of the image. Defaults to 1.0.
        
    Returns:
        A normalized image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        image = image.to(dtype=torch.get_default_dtype()) \
            if not image.is_floating_point() else image
        ratio = (new_max - new_min) / (max - min)
        image = (image - min) * ratio + new_min
        # image = torch.clamp(image, new_min, new_max)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError(f"This function has not been implemented.")
    else:
        raise TypeError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


denormalize_image = functools.partial(normalize_by_range, min=0.0, max=1.0,   new_min=0.0, new_max=255.0)
normalize_image   = functools.partial(normalize_by_range, min=0.0, max=255.0, new_min=0.0, new_max=1.0  )

# endregion


# region Operation

def add_weighted(
    image1: torch.Tensor,
    alpha : float,
    image2: torch.Tensor,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image of shape [..., C, H, W].
        alpha: The weight of the :param:`image1` elements.
        image2: The second image of same shape as :param:`image1`.
        beta: The weight of the :param:`image2` elements.
        gamma: A scalar added to each sum. Defaults to 0.0.

    Returns:
        A weighted image of shape [..., C, H, W].
    """
    assert isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape
    
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    output = output.clamp(0, bound).to(image1.dtype)
    return output

# endregion
