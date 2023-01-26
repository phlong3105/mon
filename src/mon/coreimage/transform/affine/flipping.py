#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements flipping transformations. """

from __future__ import annotations

__all__ = [
    "horizontal_flip", "horizontal_flip_image_box", "vertical_flip",
    "vertical_flip_image_box",
]

import torch

from mon.coreimage import geometry, util


def horizontal_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip an image horizontally.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        
    Returns:
        A flipped imag eof shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    image = image.clone()
    image = image.flip(-1)
    return image


def horizontal_flip_image_box(
    image: torch.Tensor,
    box  : torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip an image and a bounding box horizontally.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        box: A box of shape [N, 4] to be transformed.
        
    Returns:
        A flipped image of shape [..., C, H, W].
        A flipped box of shape [N, 4].
    """
    center = util.get_image_center4(image)
    image  = horizontal_flip(image=image)
    box    = geometry.horizontal_flip_box(box=box, image_center=center)
    return image, box

    
def vertical_flip(image: torch.Tensor) -> torch.Tensor:
    """Flip an image vertically.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        
    Returns:
        A flipped image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    image = image.clone()
    image = image.flip(-2)
    return image


def vertical_flip_image_box(
    image: torch.Tensor,
    box  : torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip image and bounding box vertically.
    
    Args:
        image: An image of shape [..., C, H, W] to be transformed.
        box: A box of shape [N, 4] to be transformed.
        
    Returns:
        A flipped image of shape [..., C, H, W].
        A flipped box of shape [N, 4].
    """
    center = util.get_image_center4(image)
    image  = vertical_flip(image=image)
    box    = geometry.vertical_flip_box(box=box, image_center=center)
    return image, box
