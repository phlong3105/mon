#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image atomic operations.

This module provides pure functions that perform a single mathematical or
structural change to the image data.
"""

__all__ = [
    "boundary_aware_prior",
    "brightness_attention_map",
    "center",
    "imgsz",
    "is_channel_first",
    "is_channel_last",
    "is_color",
    "is_grayscale",
    "is_image",
    "is_normalized",
    "num_channels",
    "pad_square",
    "pair_downsample",
    "shape",
    "split",
    "to_array",
    "to_channel_first",
    "to_channel_last",
    "to_tensor",
]

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# ==============================================================================
# VALIDATION & SANITIZATION (Integrity Checks)
# ==============================================================================

# --- Verify (Schema and range checking) ---
def is_image(image: torch.Tensor | np.ndarray) -> bool:
    """Check if the input is an image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
            
    Returns:
        True if the input is an image, otherwise False.
    """
    return (
        isinstance(image, torch.Tensor | np.ndarray)
        and (is_color(image) or is_grayscale(image))
    )


def is_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Check if an image is in channel-first format.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
            
    Returns:
        True if the image is in channel-first format, otherwise False.

    Raises:
        ValueError: If unable to determine the channel format.
    """
    # Determine tensor type and get shape
    if isinstance(image, torch.Tensor):
        shape_ = image.size()
    elif isinstance(image, np.ndarray):
        shape_ = image.shape
    else:
        raise TypeError(f"``image`` must be a numpy.ndarray or torch.Tensor, got {type(image)}.")
    
    # Check if tensor has at least 3 dimensions (batch, height/width, channels)
    if not 3 <= len(shape_) <= 4:
        raise ValueError(f"``image`` must have at least 3 dimensions, got {len(shape_)}.")
    
    # Extract dimensions
    if len(shape_) == 3:
        s0, s1, s2    = shape_
    else:
        _, s0, s1, s2 = shape_
    
    # Heuristic: Channels are typically smaller than spatial dimensions
    if (s0 < s1) and (s0 < s2):
        return True
    elif (s2 < s0) and (s2 < s1):
        return False
    else:
        raise ValueError(f"Cannot determine channel format for shape [{shape_}].")


def is_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Check if an image is in channel-last format.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
            
    Returns:
        True if the image is in channel-last format, otherwise False.
    """
    return not is_channel_first(image)


def is_color(image: torch.Tensor | np.ndarray) -> bool:
    """Check if an image is a color image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].

    Returns:
        True if the image has 3 or 4 channels, False otherwise.

    Notes:
        Assumes a color image has 3 or 4 channels (e.g., RGB or RGBA).
    """
    return num_channels(image) in [3, 4]


def is_grayscale(image: torch.Tensor | np.ndarray) -> bool:
    """Check if an image is a grayscale image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
   
    Returns:
        True if the image has 1 channel or is 2D, False otherwise.
    """
    return num_channels(image) == 1 or len(image.shape) == 2


def is_normalized(image: torch.Tensor | np.ndarray) -> bool:
    """Check if an image is normalized to range [-1.0, 1.0] or [0.0, 1.0].

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
    
    Returns:
        True if the image is normalized, False otherwise.
    
    Raises:
        TypeError: If ``image`` is neither a torch.Tensor nor a numpy.ndarray.
    """
    if isinstance(image, torch.Tensor):
        return bool(abs(torch.max(image)) <= 1.0)
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")


# --- Clean (Fixing corrupt values/nulls) ---


# ==============================================================================
# GEOMETRIC TRANSFORMATIONS (Resizing, Warping)
# ==============================================================================

# --- Analytics (Area, Perimeter, Centroid calculations) ---
def center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Extract the center coordinates of an image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
    
    Returns:
        The center of an image as (H/2, W/2).
    """
    h, w    = imgsz(image)
    center_ = [h / 2, w / 2]
    return torch.tensor(center_) if isinstance(image, torch.Tensor) else np.array(center_)


def shape(image: torch.Tensor | np.ndarray) -> tuple[int, int, int]:
    """Extract the shape of an image as (H, W, C).

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].

    Returns:
        The shape of the image as (H, W, C).
    """
    return (
        (image.shape[-2], image.shape[-1], image.shape[-3])
        if is_channel_first(image)
        else (image.shape[-3], image.shape[-2], image.shape[-1])
    )


def imgsz(image_or_size: Any, divisor: int = None) -> tuple[int, int]:
    """Extract the size of an image as (H, W).

    Args:
        image_or_size: Image as a torch.Tensor or numpy.ndarray, or size as int,
            Sequence[int].
        divisor: Divisor size for height and width.

    Returns:
        The size of the image as (H, W).

    Raises:
        TypeError: If ``image_or_size`` is not a valid type.
    """
    size = None
    if isinstance(image_or_size, list | tuple):
        if len(image_or_size) == 1:
            size = (image_or_size[0], image_or_size[0])
        elif len(image_or_size) == 2:
            size = image_or_size
        elif len(image_or_size) == 3:
            size = image_or_size[:2] if len(image_or_size) == 3 and image_or_size[0] >= image_or_size[2] else image_or_size[-2:]
    elif isinstance(image_or_size, (int, float)):
        size = (image_or_size, image_or_size)
    elif isinstance(image_or_size, torch.Tensor | np.ndarray):
        size = (
            (int(image_or_size.shape[-2]), int(image_or_size.shape[-1]))
            if is_channel_first(image_or_size)
            else (int(image_or_size.shape[-3]), int(image_or_size.shape[-2]))
        )
    else:
        raise TypeError(f"``input`` must be a torch.Tensor, numpy.ndarray, int, "
                        f"Sequence[int], str, or core.Path, got {type(image_or_size)}.")

    if divisor is not None:
        size = tuple(int(math.ceil(dim / divisor) * divisor) for dim in size)
    return size


def num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Extract the number of channels in an image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
   
    Returns:
        The number of channels in the image.
    """
    if image.ndim == 4:
        c = image.shape[1] if is_channel_first(image) else image.shape[3]
    elif image.ndim == 3:
        c = image.shape[0] if is_channel_first(image) else image.shape[2]
    elif image.ndim == 2:
        c = 1
    else:
        c = 0
    return c


# --- Metrics ---


# --- Project (Affine, Perspective, and Coordinate space transforms) ---


# --- Reshape (Resize, Crop, Padding) ---
def pad_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Pad an image to make it a square.

    Args:
        image: An RGB image as a numpy.ndarray of shape (H, W, C) with pixel
            values in the range [0, 255].
        pad_value: Padding value. Default to 0.
        
    Returns:
        Padded square image of shape (S, S, C), where S is the maximum of (H, W).
    
    Raises:
        ValueError: If ``image`` is not a 3D numpy array.
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"``image`` must be a numpy.ndarray of shape (H, W, C), "
                         f"got {image.shape} with {len(image.shape)} dimensions.")
    
    h, w, c  = image.shape
    size     = max(h, w)
    padded   = np.full((size, size, c), pad_value, dtype=image.dtype)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded


def pair_downsample(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample an image tensor into a pair to half resolution.
    
    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1].

    Returns:
        A tuple containing two downsampled images of shape (B, C, H/2, W/2).

    Notes:
        Averages diagonal pixels in non-overlapping patches:
            -------------      -------------
            | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
            | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
            -------------  =>  -------------
            | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
            | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
            -------------      -------------

    References:
        - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TypeError(f"``image`` must be a torch.Tensor of shape (B, C, H, W), "
                        f"got {type(image)} with {image.ndim} dimensions.")
    
    b, c, h, w = image.shape
    filter1    = torch.Tensor([[[[0, 0.5], [0.5, 0]]]]).to(image.dtype).to(image.device)
    filter1    = filter1.repeat(c, 1, 1, 1)
    filter2    = torch.Tensor([[[[0.5, 0], [0, 0.5]]]]).to(image.dtype).to(image.device)
    filter2    = filter2.repeat(c, 1, 1, 1)
    output1    = F.conv2d(image, filter1, stride=2, groups=c)
    output2    = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


def split(image: np.ndarray, n: int = 2) -> list[np.ndarray]:
    """Split an image into ``n`` equal parts.

    Args:
        image: An RGB image as a numpy.ndarray of shape (H, W, C) with pixel
            values in the range [0, 255].
        n: Number of parts to split the image into. Default to 2.

    Returns:
        List of ``n`` sub-images as numpy.ndarray of shape approximately (H/n, W/n, C).

    Raises:
        ValueError: If ``image`` is not a 3D numpy array.
        ValueError: If ``n`` is not a positive integer.
        ValueError: If ``n`` exceeds the total number of pixels in the image.
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"``image`` must be a numpy.ndarray of shape (H, W, C), "
                         f"got {image.shape} with {len(image.shape)} dimensions.")
    if n < 1:
        raise ValueError(f"``n`` must be a positive integer, got {n}.")

    h, w = imgsz(image)
    if n > h * w:
        raise ValueError(f"``n`` ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))

    return sub_images


# ==============================================================================
# STATISTICAL OPERATIONS (Normalization, Scaling)
# ==============================================================================

# --- Normalize (Mean/Std, Min-Max scaling) ---


# --- Standardize (Unit conversion) ---


# ==============================================================================
# CONVERSIONS
# ==============================================================================

# --- Formats ---
def to_channel_first(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to channel-first format.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
    
    Returns:
        A channel-first image as a torch.Tensor of shape (B, C, H, W) with pixel
        values in the range [0, 1] or a numpy.ndarray of shape (H, W, C) with
        pixel values in the range [0, 255].
    
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a torch.Tensor or numpy.ndarray.
    """
    if is_channel_first(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"``image``'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(2, 0, 1)     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))     # [H, W, C] -> [C, H, W]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


def to_channel_last(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an image to channel-last format.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1] or a numpy.ndarray of shape (H, W, C)
            with pixel values in the range [0, 255].
            
    Returns:
        A channel-last image as a torch.Tensor of shape (B, C, H, W) with pixel
        values in the range [0, 1] or a numpy.ndarray of shape (H, W, C) with
        pixel values in the range [0, 255].
            
    Raises:
        ValueError: If ``image`` dimensions are not 3 or 4.
        TypeError: If ``image`` is not a torch.Tensor or numpy.ndarray.
    """
    if is_channel_last(image):
        return image
    if not 3 <= image.ndim <= 4:
        raise ValueError(f"``image``'s number of dimensions must be between 3 and 4, got {image.ndim}.")
    
    if isinstance(image, torch.Tensor):
        image = image.clone()
        if image.ndim == 3:
            image = image.permute(1, 2, 0)     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    elif isinstance(image, np.ndarray):
        image = np.copy(image)  # Changed from copy.deepcopy for efficiency
        if image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))     # [C, H, W] -> [H, W, C]
        elif image.ndim == 4:
            image = np.transpose(image, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
    else:
        raise TypeError(f"``image`` must be a torch.Tensor or numpy.ndarray, got {type(image)}.")
    
    return image


# --- Types ---
def to_array(image: torch.Tensor) -> np.ndarray:
    """Convert an image from torch.Tensor to numpy.ndarray.
    
    Args:
        image: Image as a torch.Tensor of shape (B, C, H, W) with pixel values
            in the range [0.0, 1.0].
    
    Returns:
        Image as a numpy.ndarray of shape (H, W, C) with pixel values in the
        range [0, 255].
    
    Raises:
        TypeError: If ``image`` is not a torch.Tensor or does not have 4 dimensions.
        
    Notes:
        image = (tensor.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TypeError(f"``image`` must be a torch.Tensor of shape (B, C, H, W), "
                        f"got {type(image)} with {image.ndim} dimensions.")
    
    image = (image.squeeze().detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    image = np.clip(image * 255, 0, 255).astype("uint8")
    return image
    

def to_tensor(image: np.ndarray, normalize: bool = False) -> torch.Tensor:
    """Convert an image from numpy.ndarray to torch.Tensor.

    Args:
        image: Image as a numpy.ndarray of shape (H, W, C) with pixel values in
            the range [0, 255].
        normalize: If True, normalizes pixel values to [0.0, 1.0]. Default to False.

    Returns:
        Image as a torch.Tensor of shape (1, C, H, W) with pixel values in the
        range [0, 1] if ``normalize`` is True, else in [0, 255].
    
    Raises:
        TypeError: If ``image`` is not a 3D numpy.array.
        
    Notes:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0).to(device)
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise TypeError(f"``image`` must be a numpy.ndarray of shape (H, W, C), "
                        f"got {type(image)} with {len(image.shape)} dimensions.")
    
    if normalize:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255.0).unsqueeze(0)
    else:
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().unsqueeze(0)
    return image
