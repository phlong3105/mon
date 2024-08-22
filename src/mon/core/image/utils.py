#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Utilities.

This module implements utility functions for image processing.
"""

from __future__ import annotations

__all__ = [
    "check_image_size",
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
    "parse_hw",
    "to_3d_image",
    "to_4d_image",
    "to_5d_image",
    "to_channel_first_image",
    "to_channel_last_image",
    "to_image_nparray",
    "to_image_tensor",
    "to_list_of_3d_image",
]

import copy
import math
from typing import Any

import numpy as np
import torch

from mon.core.rich import error_console
from mon.core.typing import _size_any_t


# region Assert

def is_channel_first_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in the channel-first format. We assume
    that if the first dimension is the smallest.
    """
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
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
    return not is_channel_first_image(image)


def is_color_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a color image. It is assumed that the
    image has ``3`` or ``4`` channels.
    """
    if get_image_num_channels(image) in [3, 4]:
        return True
    return False


def is_gray_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a gray image. It is assumed that the
    image has one channel.
    """
    if get_image_num_channels(image) in [1] or len(image.shape) == 2:
        return True
    return False


def is_color_or_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is a color or gray image.
    """
    return is_color_image(image) or is_gray_image(image)


def is_image(image: torch.Tensor, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly ``[0.0, 1.0]`` for
    :obj:`float` or `[0, 2 ** bits]` for :obj:`int`.

    Args:
        image: Image tensor to evaluate.
        bits: The image bits. The default checks if given :obj:`int` input
            image is an 8-bit image `[0-255]` or not.

    Raises:
        TypeException: if all the input tensor has not
        1) a shape `[3, H, W]`,
        2) ``[0.0, 1.0]`` for :obj:`float` or ``[0, 255]`` for :obj:`int`,
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
    c = get_image_num_channels(image)
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
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")


def is_one_hot_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is one-hot encoded."""
    c = get_image_num_channels(image)
    if c > 1:
        return True
    return False


def check_image_size(size: list[int], stride: int = 32) -> int:
    """If the input :obj:`size` isn't a multiple of the :obj:`stride`,
    then the image size is updated to the next multiple of the stride.
    
    Args:
        size: An image's size.
        stride: The stride of a network. Default: ``32``.
    
    Returns:
        A new size of the image.
    """
    size     = parse_hw(size)
    size     = size[0]
    new_size = make_imgsz_divisible(size, int(stride))
    if new_size != size:
        error_console.log("WARNING: image_size %g must be multiple of max "
                          "stride %g, updating to %g" % (size, stride, new_size))
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
    
    if is_channel_first_image(image):
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
    return get_channel(image, index=0, keep_dim=True)


def get_last_channel(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the first channel of an image."""
    return get_channel(image, index=-1, keep_dim=True)


def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An image in channel-last or channel-first format.
    """
    if image.ndim == 4:
        if is_channel_first_image(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
    elif image.ndim == 3:
        if is_channel_first_image(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
    elif image.ndim == 2:
        c = 1
    else:
        # error_console.log(
        #     f":obj:`image`'s number of dimensions must be between ``2`` and ``4``, "
        #     f"but got {input.ndim}."
        # )
        c = 0
    return c


def get_image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as `(x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    `(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        image: An image in channel-last or channel-first format.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")


def get_image_size(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height and width value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first_image(image):
        return [image.shape[-2], image.shape[-1]]
    else:
        return [image.shape[-3], image.shape[-2]]


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An image.
    """
    if is_channel_first_image(image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]

# endregion


# region Convert

def to_3d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D or 4D image to a 3D.

    Args:
        image: An image in channel-first format.

    Return:
        A 3D image in channel-first format.
    """
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``4``, but got {image.ndim}.")
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
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


def to_list_of_3d_image(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :obj:`list` of 3D images.
   
    Args:
        image: An image of arbitrary type.
        
    Return:
        A :obj:`list` of 3D images.
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


def to_4d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 5D, list of 3D, and list of 4D images to 4D.

    Args:
        image: A 2D, 3D, 5D, list of 3D, and list of 4D images in channel-first format.

    Return:
        A 4D image in channel-first format.
    """
    if isinstance(image, (torch.Tensor, np.ndarray)) and not 2 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``5``, but got {image.ndim}.")
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
        raise TypeError(f"`image` must be a `numpy.ndarray`, `torch.Tensor`, "
                        f"or a `list` of either of them, but got {type(image)}.")
    return image


def to_5d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D, 3D, 4D, or 6D image to a 5D.
    
    Args:
        image: An tensor in channel-first format.

    Return:
        A 5D image in channel-first format.
    """
    if not 2 <= image.ndim <= 6:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``6``, but got {image.ndim}.")
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
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


def to_channel_first_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format.
    
    Args:
        image: An image in channel-last or channel-first format.
    
    Returns:
        An image in channel-first format.
    """
    if is_channel_first_image(image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
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
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    return image


def to_channel_last_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format.

    Args:
        image: An image in channel-last or channel-first format.

    Returns:
        A image in channel-last format.
    """
    if is_channel_last_image(image):
        return image
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
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
            f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
            f"but got {type(image)}."
        )
    return image


def to_image_nparray(
    image      : torch.Tensor | np.ndarray,
    keepdim    : bool = False,
    denormalize: bool = False,
) -> np.ndarray:
    """Convert an image to :obj:`numpy.ndarray`.
    
    Args:
        image: An image.
        keepdim: If `True`, keep the original shape. If ``False``, convert it to
            a 3D shape. Default: ``True``.
        denormalize: If ``True``, convert image to ``[0, 255]``. Default: ``True``.

    Returns:
        An :obj:`numpy.ndarray` image.
    """
    from mon.core.image.photometry import denormalize_image
    
    if not 3 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        image = image.detach()
        image = image.cpu().numpy()
    image = denormalize_image(image).astype(np.uint8) if denormalize else image
    image = to_channel_last_image(image)
    if not keepdim:
        image = to_3d_image(image)
    return image


def to_image_tensor(
    image    : torch.Tensor | np.ndarray,
    keepdim  : bool = False,
    normalize: bool = False,
    device   : Any  = None,
) -> torch.Tensor:
    """Convert an image from :obj:`PIL.Image` or :obj:`numpy.ndarray` to
    :obj:`torch.Tensor`. Optionally, convert :obj:`image` to channel-first
    format and normalize it.
    
    Args:
        image: An image in channel-last or channel-first format.
        keepdim: If ``True``, keep the original shape. If ``False``, convert it
            to a 4D shape. Default: ``True``.
        normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
            Default: ``False``.
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        
    Returns:
        A :obj:`torch.Tensor` image.
    """
    from mon.core.image.photometry import normalize_image
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(image)}.")
    image = to_channel_first_image(image)
    if not keepdim:
        image = to_4d_image(image)
    image = normalize_image(image) if normalize else image
    # Place in memory
    image = image.contiguous()
    if device:
        image = image.to(device)
    return image

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
    """Casts a size object to the standard ``[H, W]``.

    Args:
        size: A size of an image, windows, or kernels, etc.

    Returns:
        A size in ``[H, W]`` format.
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
