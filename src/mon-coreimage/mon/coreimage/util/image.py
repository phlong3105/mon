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

import functools
from typing import TypeAlias

import numpy as np
import PIL.Image
import torch
from torchvision.transforms import functional

from mon.coreimage.typing import FloatAnyT, Image, Int2T, Int3T, Ints
from mon.coreimage.util import tensor
from mon.foundation import error_console, math


# region Conversion

def correct_image_dimension(image: Image) -> Image:
    """Corrects the dimensionality of :param:`image` to the correct format used
    by :mod:`matplotlib` and :mod:`cv2`.
    
    Args:
        image: Image to be corrected.
    
    Returns:
        Corrected image ready to be used.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 4
    assert is_channel_last(image)
   
    image = image.clone()
    if image.ndim == 2:
        pass
    elif image.ndim == 3:
        if image[-1] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze(-1)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        image = image.permute(0, 2, 3, 1)
        if image[-1] == 1:
            image = image.squeeze(-1)
    return image


def correct_tensor_dimension(image: torch.Tensor) -> Image:
    """Corrects the dimensionality of :param:`image` to the correct format used
    by :mod:`torch`. It should be a :class:`torch.Tensor` in channel-first
    format and has shape [..., 1 or B, C, H, W]

    Args:
        image: Image to be corrected.

    Returns:
        Corrected image ready to be used.
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


def to_channel_first(image: Image) -> Image:
    """Converts :param:`image` to channel first format.
    
    Args:
        image: Image to be converted.
    
    Returns:
        A channel first image.
    """
    if is_channel_first(image):
        return image

    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 5
    image = image.clone()
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(2, 0, 1)
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)
        elif image.ndim == 5:
            image = image.permute(0, 1, 4, 2, 3)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 4, 2, 3))
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def to_channel_last(image: Image) -> Image:
    """Converts :param:`image` to channel last format.

    Args:
        image: Image to be converted.

    Returns:
        A channel last image.
    """
    if is_channel_last(image):
        return image

    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim <= 5
    image = image.clone()
    
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.permute(1, 2, 0)
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)
        elif image.ndim == 5:
            image = image.permute(0, 1, 3, 4, 2)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))
        elif image.ndim == 5:
            image = np.transpose(image, (0, 1, 3, 4, 2))
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )
    return image


def to_image(
    image      : torch.Tensor,
    keepdim    : bool = True,
    denormalize: bool = False,
) -> np.ndarray:
    """Converts :param:`image` from :class:`torch.Tensor` to
    :class:`np.ndarray`.
    
    Args:
        image: Image of shape [..., C, H, W] to be converted.
        keepdim: If True, the function will keep the dimensions of the input
            tensor. Defaults to True.
        denormalize: If True, the image will be denormalized to [0, 255].
            Defaults to False.
        
    Returns:
        An image in :class:`np.ndarray`.
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


def to_pil_image(image: Image) -> PIL.Image:
    """Converts :param:`image` from :class:`torch.Tensor` or
    :class:`np.ndarray` to :class:`PIL.Image`.
    
    Args:
        image: Image to be converted.
    
    Returns:
        A :class:`PIL.Image` image.
    """
    if isinstance(image, torch.Tensor):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return functional.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def to_tensor(
    image    : Image | PIL.Image,
    keepdim  : bool = True,
    normalize: bool = False,
) -> torch.Tensor:
    """Converts :param:`image` from :class:`PIL.Image` or :class:`np.ndarray`
    to :class:`torch.Tensor`. Optionally, convert :param:`image` to
    channel-first format and normalize it.
    
    Args:
        image: Image to be converted.
        keepdim: If True, the channel dimension will be kept. If False unsqueeze
            the image to match the shape [..., C, H, W]. Defaults to True
        normalize: If True, normalize the image to [0, 1]. Defaults to False
    
    Returns:
        A :class:`torch.Tensor` image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray) \
           or functional._is_pil_image(image)
    if isinstance(image, torch.Tensor | np.ndarray):
        assert 2 <= image.ndim <= 4
    
    image = image.clone()
    # Handle :class:`PIL.Image`
    if functional._is_pil_image(image):
        mode           = image.mode
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        image          = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        if mode == "1":
            image = 255 * image
    
    # Handle :class:`np.ndarray`
    if functional._is_numpy(image):
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
    """If the input :param:`size` is not a multiple of the :param:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: The size of the image.
        stride: The stride of the network. Defaults to 32.
    
    Returns:
        A new size of the image.
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:    # [C, H, W]
            size = size[1]
        elif len(size) == 2:  # [H, W]
            size = size[0]
        
    new_size = math.make_divisible(size, int(stride))  # ceil gs-multiple
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


def get_image_center(image: Image) -> Image:
    """Returns the coordinates of the center of :param:`image` as (x=h/2,
    y=w/2).
    
    Args:
        image: Image in arbitrary type.
    
    Returns:
        The center values of the image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    h, w = get_image_hw(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def get_image_center4(image: Image) -> Image:
    """Returns the coordinates of the center of :param:`image` as (x=h/2,
    y=w/2, x=h/2, y=w/2).
    
    Args:
        image: Image in arbitrary type.
    
    Returns:
        The center values of the image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    h, w = get_image_hw(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def get_image_hw(image: Image) -> Int2T :
    """Returns height and width value of :param:`image`.
    
    Args:
        image: Image in arbitrary format.
    
    Returns:
        The height and width of the image.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    # [.., C, H, W]
    if is_channel_first(image):
        return image.shape[-2], image.shape[-1]
    # [.., H, W, C]
    else:
        return image.shape[-3], image.shape[-2]
    

def get_image_shape(image: Image) -> Int3T :
    """Returns channel, height, and width value of :param:`image`.
    
    Args:
        image: Image in arbitrary format.
    
    Returns:
        The shape of the image as [C, H, W].
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


def get_num_channels(image: Image) -> int:
    """Returns the number of channels of :param:`image`.

    Args:
        image: Image to get the number of channels from.

    Returns:
        The number of channels of :param:`image`.
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
    """Casts :param:`size` of arbitrary format into standard [H, W].

    Args:
        size: The size of the image, windows, kernels, etc.

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

def is_channel_first(image: Image) -> bool:
    """Returns True if :param:`image` is currently in channel first format. It
    is assumed that if the first dimension is the smallest, then it's channel
    first.
    
    Args:
        image: Image to be checked.

    Returns:
        True if :param:`image` is in channel fist format.
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


def is_channel_last(image: Image) -> bool:
    """Returns True if :param:`image` is currently in channel last format
    (default). It is assumed that if the first dimension is the smallest, then
    it's channel first.
    
    Args:
        image: Image to be checked.

    Returns:
        True if :param:`image` is in channel last format.
    """
    return not is_channel_first(image=image)


def is_color_image(image: Image) -> bool:
    """Returns True if :param:`image` is currently a colored image. It is
    assumed that the image has 3 or 4 channels.
    """
    if get_num_channels(image) in [3, 4]:
        return True
    return False


def is_integer_image(image: Image) -> bool:
    """Returns True if the given :param:`image` is integer-encoded."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    c = get_num_channels(image=image)
    if c == 1:
        return True
    return False


def is_normalized(image: Image) -> bool:
    """Returns True if the given :param:`image` is normalized."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise ValueError(
            f":param:`image` must be `torch.Tensor` or a `numpy.ndarray`. "
            f"But got: {type(image)}."
        )


def is_one_hot_image(image: Image) -> bool:
    """Returns True if the given :param:`image` is one-hot encoded."""
    assert isinstance(image, torch.Tensor | np.ndarray)
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


def denormalize_mean_std(
    image: torch.Tensor,
    mean : FloatAnyT = (0.485, 0.456, 0.406),
    std  : FloatAnyT = (0.229, 0.224, 0.225),
    eps  : float     = 1e-6,
) -> torch.Tensor:
    """Denormalizes :param:`image` with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: Image of shape [..., C, H, W] to be adjusted, where ... means it
            can have an arbitrary number of leading dimensions.
        mean: Sequence of means for each channel. Defaults to
            (0.485, 0.456, 0.406).
        std: Sequence of standard deviations for each channel. Defaults to
            (0.229, 0.224, 0.225).
        eps: Avoid zero division. Defaults to 1e-6.
        
    Returns:
        Denormalized image with same size as input.
    """
    assert isinstance(image, torch.Tensor)
    assert 3 <= image.ndim
    image = image.clone()
    
    if not image.is_floating_point():
        image = image.to(dtype=torch.get_default_dtype())
    
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
    std_inv  = std_inv.view(-1, 1, 1)  if std_inv.ndim == 1  else std_inv
    mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
    image.sub_(mean_inv).div_(std_inv)
    return image


def normalize_mean_std(
    image: torch.Tensor,
    mean : FloatAnyT = (0.485, 0.456, 0.406),
    std  : FloatAnyT = (0.229, 0.224, 0.225),
    eps  : float     = 1e-6,
) -> torch.Tensor:
    """Normalizes :param:`image` with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image: Image of shape [..., C, H, W] to be adjusted, where ... means it
            can have an arbitrary number of leading dimensions.
        mean: Sequence of means for each channel. Defaults to
            (0.485, 0.456, 0.406).
        std: Sequence of standard deviations for each channel. Defaults to
            (0.229, 0.224, 0.225).
        eps: Avoid zero division. Defaults to 1e-6.
        
    Returns:
        Normalized image with same size as input.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    image = image.clone()
    image = image.to(dtype=torch.float32)
    
    if not image.is_floating_point():
        image = image.to(dtype=torch.get_default_dtype())
        
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
    std  += eps
    
    mean  = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
    std   = std.view(-1, 1, 1)  if std.ndim == 1  else std
    image.sub_(mean).div_(std)
    return image


def normalize_by_range(
    image  : torch.Tensor,
    min    : float        = 0.0,
    max    : float        = 255.0,
    new_min: float        = 0.0,
    new_max: float        = 1.0,
) -> torch.Tensor:
    """Normalizes :param:`image` from range [:param:`min`, :param:`max`] to
    [:param:`new_min`, :param:`new_max`].
    
    Args:
        image: Image of shape [..., C, H, W] to be adjusted, where ... means it
            can have an arbitrary number of leading dimensions.
        min: Current minimum pixel value of the image. Defaults to 0.0.
        max: Current maximum pixel value of the image. Defaults to 255.0.
        new_min: New minimum pixel value of the image. Defaults to 0.0.
        new_max: New minimum pixel value of the image. Defaults to 1.0.
        
    Returns:
        Normalized image with same size as input.
    """
    assert isinstance(image, torch.Tensor | np.ndarray)
    assert 3 <= image.ndim
    image = image.clone()

    if not image.is_floating_point():
        image = image.to(dtype=torch.get_default_dtype())
    
    ratio = (new_max - new_min) / (max - min)
    image = (image - min) * ratio + new_min
    # image = torch.clamp(image, new_min, new_max)
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
    """Calculates the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: First image of shape [..., C, H, W].
        alpha: Weight of the :param:`image1` elements.
        image2: Second image of same shape as :param:`image1`.
        beta: Weight of the :param:`image2` elements.
        gamma: Scalar added to each sum. Defaults to 0.0.

    Returns:
        Weighted image of shape [..., C, H, W].
    """
    assert isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor)
    assert image1.shape == image2.shape
    
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    output = output.clamp(0, bound).to(image1.dtype)
    return output

# endregion
