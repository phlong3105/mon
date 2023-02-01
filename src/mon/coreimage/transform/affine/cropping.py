#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements cropping transformations."""

from __future__ import annotations

__all__ = [
    "center_crop", "crop", "crop_tblr", "crop_zero_region", "five_crop",
    "ten_crop",
]

import torch

from mon.coreimage import util
from mon.coreimage.transform.affine import base, flipping
from mon.coreimage.typing import Ints


def center_crop(image: torch.Tensor, output_size: Ints) -> torch.Tensor:
    """Crop an image at the center. If the image size is smaller than the output
    size along any edge, the image is padded with 0 and then center cropped.

    Args:
        image: An image of shape [..., C, H, W] to be cropped,.
        output_size: An output size of the crop. If size is an int instead of
            sequence like (h, w), a square crop (size, size) is made. If
            provided a sequence of length 1, it will be interpreted as (size[0],
            size[0]).
        
    Returns:
        A cropped image of shape [..., C, H, W].
    """
    output_size      = util.to_size(output_size)
    image_h, image_w = util.get_image_size(image)
    crop_h,  crop_w  = output_size
    image            = image.clone()
    if crop_w > image_w or crop_h > image_h:
        padding_ltrb = [
            (crop_w - image_w)     // 2 if crop_w > image_w else 0,
            (crop_h - image_h)     // 2 if crop_h > image_h else 0,
            (crop_w - image_w + 1) // 2 if crop_w > image_w else 0,
            (crop_h - image_h + 1) // 2 if crop_h > image_h else 0,
        ]
        image = base.pad(image=image, padding=padding_ltrb, fill=0)
        _, image_h, image_w = util.get_image_size(image)
        if crop_w == image_w and crop_h == image_h:
            return image

    crop_top  = int(round((image_h - crop_h) / 2.0))
    crop_left = int(round((image_w - crop_w) / 2.0))
    image     = crop(
        image   = image,
        top     = crop_top,
        left    = crop_left,
        height  = crop_h,
        width   = crop_w,
    )
    return image


def crop_tblr(
    image  : torch.Tensor,
    top    : int,
    bottom : int,
    left   : int,
    right  : int,
) -> torch.Tensor:
    """Crop an image with top + bottom + left + right value.
    
    Args:
        image: An image of shape [..., C, H, W].
        top: The top padding.
        bottom: The bottom padding.
        left: The left padding.
        right: The right padding.
        
    Returns:
        A cropped image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    bottom = -bottom if bottom > 0 else bottom
    right  = -right  if right  > 0 else right
    image  = image.clone()
    image  = image[..., top:bottom, left:right]
    return image


def crop(
    image  : torch.Tensor,
    top    : int,
    left   : int,
    height : int,
    width  : int,
) -> torch.Tensor:
    """Crop an image at specified location and output size.
    
    Args:
        image: An image of shape [..., C, H, W].
        top: The Vertical component of the top left corner of the crop bbox.
        left: The Horizontal component of the top left corner of the crop bbox.
        height: The height of the crop bbox.
        width: The width of the crop bbox.
        
    Returns:
        A cropped image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    h, w   = util.get_image_size(image)
    right  = left + width
    bottom = top  + height
    image  = image.clone()
    
    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left, 0), max(-top, 0), max(right - w, 0), max(bottom - h, 0)
        ]
        return base.pad(
            image        = image[..., max(top, 0) : bottom, max(left, 0) : right],
            padding      = padding_ltrb,
            fill         = 0,
            padding_mode = "constant",
        )
    image = image[..., top:bottom, left:right]
    return image


def crop_zero_region(image: torch.Tensor) -> torch.Tensor:
    """Crop the zero regions around the non-zero region in an image.
    
    Args:
        image: An image of shape [C, H, W] with zeros background.
        
    Returns:
        A cropped image of shape [C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    if util.is_channel_last(image):
        cols       = torch.any(image, dim=0)
        rows       = torch.any(image, dim=1)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cols       = torch.any(image, dim=1)
        rows       = torch.any(image, dim=2)
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        image      = image[:, ymin:ymax + 1, xmin:xmax + 1]
    return image


def five_crop(
    image: torch.Tensor, size: Ints
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Crop an image into four corners, and the central crop.
    
    Notes:
        This transform returns a tuple of images and there may be a mismatch in
        the number of inputs and targets your `Dataset` returns.
    
    Args:
        image: An image of shape [..., C, H, W] to be cropped.
        size: An output size of the crop. If size is an int instead of sequence
            like (h, w), a square crop (size, size) is made. If provided a
            sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
        a tuple of corresponding top left, top right, bottom left, bottom right
        and center crop.
    """
    size = util.to_size(size)
    assert isinstance(size, list | tuple) and len(size) == 2

    image_h, image_w = util.get_image_size(image)
    crop_h,  crop_w  = size
    if crop_w > image_w or crop_h > image_h:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_h, image_w)))

    tl = crop(image, 0, 0, crop_h, crop_w)
    tr = crop(image, 0, image_w - crop_w, crop_h, crop_w)
    bl = crop(image, image_h - crop_h, 0, crop_h, crop_w)
    br = crop(image, image_h - crop_h, image_w - crop_w, crop_h, crop_w)

    center = center_crop(image, [crop_h, crop_w])
    return tl, tr, bl, br, center


def ten_crop(
    image: torch.Tensor,
    size : Ints,
    vflip: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Generate ten cropped images from the given image. Crop the given image
    into four corners, and the central crop plus the flipped version of these
    (horizontal flipping is used by default).
   
    Notes:
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your `Dataset` returns.

    Args:
        image: An image of shape [..., C, H, W] to be cropped.
        size: An output size of the crop. If size is an int instead of sequence
            like (h, w), a square crop (size, size) is made. If provided a
            sequence of length 1, it will be interpreted as (size[0], size[0]).
        vflip: Use vertical flipping instead of horizontal. Defaults to False.

    Returns:
        Corresponding top left, top right, bottom left, bottom right and center
            crop and same for the flipped image.
    """
    size = util.to_size(size)
    assert isinstance(size, list | tuple) and len(size) == 2
    first_five = five_crop(image, size)
    if vflip:
        image = flipping.flip_vertical(image=image)
    else:
        image = flipping.flip_horizontal(image=image)
    second_five = five_crop(image, size)
    return first_five + second_five
