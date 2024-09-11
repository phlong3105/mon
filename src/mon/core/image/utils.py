#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Utilities.

This module implements utility functions for image processing.
"""

from __future__ import annotations

__all__ = [
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "add_weighted",
    "blend_images",
    "depth_map_to_color",
    "get_image_center",
    "get_image_center4",
    "get_image_channel",
    "get_image_num_channels",
    "get_image_shape",
    "get_image_size",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
    "is_channel_first_image",
    "is_channel_last_image",
    "is_color_image",
    "is_gray_image",
    "is_image",
    "is_integer_image",
    "is_normalized_image",
    "label_map_color_to_id",
    "label_map_id_to_color",
    "label_map_id_to_one_hot",
    "label_map_id_to_train_id",
    "label_map_one_hot_to_id",
    "to_2d_image",
    "to_3d_image",
    "to_4d_image",
    "to_channel_first_image",
    "to_channel_last_image",
    "to_image_nparray",
    "to_image_tensor",
    "to_list_of_3d_image",
]

import copy
import math
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# region Assertion

def is_channel_first_image(image: torch.Tensor | np.ndarray) -> bool:
    """Return ``True`` if an image is in channel-first format. We assume
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
    """Return ``True`` if an image is in channel-first format."""
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
    """Return ``True`` if an image is a color or gray image."""
    return is_color_image(image) or is_gray_image(image)


def is_image(image: torch.Tensor, bits: int = 8) -> bool:
    """Check whether an image tensor is ranged properly ``[0.0, 1.0]`` for
    :obj:`float` or ``[0, 2 ** bits]`` for :obj:`int`.

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
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region Accessing

def get_image_channel(
    image   : torch.Tensor | np.ndarray,
    index   : int | Sequence[int],
    keep_dim: bool = True,
) -> torch.Tensor | np.ndarray:
    """Return a channel of an image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        index: The channel's index.
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
    

def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Return the number of channels of an image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
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
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def get_image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Return the center of a given image specified as
    `(x=h/2, y=w/2, x=h/2, y=w/2)`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    h, w = get_image_size(image)
    if isinstance(image, torch.Tensor):
        return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    elif isinstance(image, np.ndarray):
        return np.array([h / 2, w / 2, h / 2, w / 2])
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def get_image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Return height, width, and channel value of an image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
    """
    if is_channel_first_image(image):
        return [image.shape[-2], image.shape[-1], image.shape[-3]]
    else:
        return [image.shape[-3], image.shape[-2], image.shape[-1]]


def get_image_size(
    input  : torch.Tensor | np.ndarray | int | Sequence[int],
    divisor: int = None,
) -> tuple[int, int]:
    """Return height and width value of an image in the ``[H, W]`` format.
    
    Args:
        input: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
            - A size of an image, windows, or kernels, etc.
        divisor: The divisor. Default: ``None``.
        
    Returns:
        A size in ``[H, W]`` format.
    """
    # Get raw size
    if isinstance(input, list | tuple):
        if len(input) == 3:
            if input[0] >= input[3]:
                size = input[0:2]
            else:
                size = input[1:3]
        elif len(input) == 2:
            size = input
        elif len(input) == 1:
            size = (input[0], input[0])
        else:
            raise ValueError(f"`input` must be a `list` of length in range "
                             f"``[1, 3]``, but got {input}.")
    elif isinstance(input, int | float):
        size = (input, input)
    elif isinstance(input, torch.Tensor | np.ndarray):
        if is_channel_first_image(input):
            size = (input.shape[-2], input.shape[-1])
        else:
            size = (input.shape[-3], input.shape[-2])
    else:
        raise TypeError(f"`input` must be a `torch.Tensor`, `numpy.ndarray`, "
                        f"or a `list` of `int`, but got {type(input)}.")
    
    # Divisible
    if divisor:
        h, w  = size
        new_h = int(math.ceil(h / divisor) * divisor)
        new_w = int(math.ceil(w / divisor) * divisor)
        size  = (new_h, new_w)
    return size


# endregion


# region Combination

def add_weighted(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        image2: The same as :obj:`image1`.
        alpha: The weight of the :obj:`image1` elements.
        beta: The weight of the :obj:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same shape, "
                         f"but got {image1.shape} != {image2.shape}.")
    if type(image1) is not type(image2):
        raise ValueError(f"`image1` and `image2` must have the same type, "
                         f"but got {type(image1)} != {type(image2)}.")
    
    output = image1 * alpha + image2 * beta + gamma
    bound  = 1.0 if is_normalized_image(image1) else 255.0
    
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound).astype(image1.dtype)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(input)}.")
    return output


def blend_images(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :obj:`image1` * alpha + :obj:`image2` * beta + gamma

    Args:
        image1: A source image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        image2: An overlay image that we want to blend on top of :obj:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.
    
    Returns:
        A blended image.
    """
    return add_weighted(
        image1 = image2,
        image2 = image1,
        alpha  = alpha,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion


# region Conversion

def depth_map_to_color(
    depth_map: np.ndarray,
    color_map: int = cv2.COLORMAP_JET,
    use_rgb  : bool = False,
) -> np.ndarray:
    """Convert depth map to color-coded images.
    
    Args:
        depth_map: A depth map of type :obj:`numpy.ndarray` in ``[H, W, 1]``
            format.
        color_map: A color map for the depth map. Default: ``cv2.COLORMAP_JET``.
        use_rgb: If ``True``, convert the heatmap to RGB format.
            Default: ``False``.
    """
    if is_normalized_image(depth_map):
        depth_map = np.uint8(255 * depth_map)
    depth_map = cv2.applyColorMap(np.uint8(255 * depth_map), color_map)
    if use_rgb:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    return depth_map
    

def label_map_id_to_train_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from IDs to train IDs.
    
    Args:
        label_map: An IDs label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels.
    """
    id2train_id = classlabels.id2train_id
    h, w        = get_image_size(label_map)
    label_ids   = np.zeros((h, w), dtype=np.uint8)
    label_map   = to_2d_image(label_map)
    for id, train_id in id2train_id.items():
        label_ids[label_map == id] = train_id
    label_ids   = np.expand_dims(label_ids, axis=-1)
    return label_ids
 

def label_map_id_to_color(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from label IDs to color-coded.
    
    Args:
        label_map: An IDs label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color  = classlabels.id2color
    h, w      = get_image_size(label_map)
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    label_map = to_2d_image(label_map)
    for id, color in id2color.items():
        color_map[label_map == id] = color
    return color_map


def label_map_color_to_id(
    label_map  : np.ndarray,
    classlabels: "ClassLabels",
) -> np.ndarray:
    """Convert label map from color-coded to label IDS.

    Args:
        label_map: A color-coded label map of type :obj:`numpy.ndarray` in
            ``[H, W, C]`` format.
        classlabels: A list of class-labels, each has predefined color.
    """
    id2color  = classlabels.id2color
    h, w      = get_image_size(label_map)
    label_ids = np.zeros((h, w), dtype=np.uint8)
    for id, color in id2color.items():
        label_ids[np.all(label_map == color, axis=-1)] = id
    label_ids = np.expand_dims(label_ids, axis=-1)
    return label_ids


def label_map_id_to_one_hot(
    label_map  : torch.Tensor | np.ndarray,
    num_classes: int           = None,
    classlabels: "ClassLabels" = None,
) ->torch.Tensor | np.ndarray:
    """Convert label map from label IDs to one-hot encoded.
    
    Args:
        label_map: An IDs label map of type:
            - :obj:`torch.Tensor` in ``[B, 1, H, W]`` format.
            - :obj:`numpy.ndarray` in ``[H, W, 1]`` format.
        num_classes: The number of classes in the label map.
        classlabels: A list of class-labels.
    """
    if num_classes is None and classlabels is None:
        raise ValueError("Either `num_classes` or `classlabels` must be "
                         "provided.")
    
    num_classes = num_classes or classlabels.num_trainable_classes
    if isinstance(label_map, torch.Tensor):
        label_map = to_3d_image(label_map).long()
        one_hot   = F.one_hot(label_map, num_classes)
        one_hot   = to_channel_first_image(one_hot).contiguous()
    elif isinstance(label_map, np.ndarray):
        label_map = to_2d_image(label_map)
        one_hot   = np.eye(num_classes)[label_map]
    else:
        raise TypeError(f"`label_map` must be a `numpy.ndarray` or "
                        f"`torch.Tensor`, but got {type(label_map)}.")
    return one_hot


def label_map_one_hot_to_id(
    label_map: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Convert label map from one-hot encoded to label IDs.
    
    Args:
        label_map: A one-hot encoded label map of type:
            - :obj:`torch.Tensor` in ``[B, num_classes, H, W]`` format.
            - :obj:`numpy.ndarray` in ``[H, W, num_classes]`` format.
    """
    if isinstance(label_map, torch.Tensor):
        label_map = torch.argmax(label_map, dim=-1, keepdim=True)
    elif isinstance(label_map, np.ndarray):
        label_map = np.argmax(label_map, axis=-1, keepdims=True)
    else:
        raise TypeError(f"`label_map` must be a `numpy.ndarray` or "
                        f"`torch.Tensor`, but got {type(label_map)}.")
    return label_map


def to_2d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 3D or 4D image to a 2D."""
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``3`` and ``4``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:  # 1HW -> HW
            image = image.squeeze(dim=0)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 1:  # 11HW -> HW
            image = image.squeeze(dim=0)
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:  # HW1 -> HW
            image = np.squeeze(image, axis=-1)
        elif image.ndim == 4 and image.shape[0] == 1 and image.shape[3] == 1:  # 1HW1 -> HW
            image = np.squeeze(image, axis=0)
            image = np.squeeze(image, axis=-1)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_3d_image(image: Any) -> torch.Tensor | np.ndarray:
    """Convert a 2D or 4D image to a 3D."""
    if not 2 <= image.ndim <= 4:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``4``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 1HW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 4 and image.shape[1] == 1:  # B1HW -> BHW
            image = image.squeeze(dim=1)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1CHW -> CHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> HW1
            image = np.expand_dims(image, axis=-1)
        elif image.ndim == 4 and image.shape[0] == 1:  # 1HWC -> HWC
            image = np.squeeze(image, axis=0)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_list_of_3d_image(image: Any) -> list[torch.Tensor | np.ndarray]:
    """Convert arbitrary input to a :obj:`list` of 3D images."""
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
    """Convert a 2D, 3D, 5D, list of 3D, and list of 4D images to 4D."""
    if isinstance(image, (torch.Tensor, np.ndarray)) and not 2 <= image.ndim <= 5:
        raise ValueError(f"`image`'s number of dimensions must be between "
                         f"``2`` and ``5``, but got {image.ndim}.")
    if isinstance(image, torch.Tensor):
        if image.ndim == 2:  # HW -> 11HW
            image = image.unsqueeze(dim=0)
            image = image.unsqueeze(dim=0)
        elif image.ndim == 3:  # CHW -> 1CHW
            image = image.unsqueeze(dim=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1BCHW -> BCHW
            image = image.squeeze(dim=0)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:  # HW -> 1HW1
            image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:  # HWC -> 1HWC
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 5 and image.shape[0] == 1:  # 1BHWC -> BHWC
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


def to_channel_first_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-first format."""
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
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    return image


def to_channel_last_image(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to the channel-last format."""
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
            f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
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
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        keepdim: If `True`, keep the original shape. If ``False``, convert it to
            a 3D shape. Default: ``True``.
        denormalize: If ``True``, convert image to ``[0, 255]``. Default: ``True``.

    Returns:
        An image of type :obj:`numpy.ndarray`.
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
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        keepdim: If ``True``, keep the original shape. If ``False``, convert it
            to a 4D shape. Default: ``True``.
        normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
            Default: ``False``.
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        
    Returns:
        A image of type :obj:`torch.Tensor`.
    """
    from mon.core.image.photometry import normalize_image
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).contiguous()
    elif isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
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


# region Gradient

def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        return image_local_stddev(image, self.patch_size, self.eps)
    
# endregion
