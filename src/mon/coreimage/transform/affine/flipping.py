#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements flipping transformations. """

from __future__ import annotations

__all__ = [
    "flip_horizontal", "flip_image_bbox_horizontal", "flip_image_bbox_vertical",
    "flip_vertical",
]

import torch

from mon.coreimage import geometry, util


def flip_horizontal(image: torch.Tensor) -> torch.Tensor:
    """Flip an image horizontally.
    
    Args:
        image: An image of shape [..., C, H, W].
        
    Returns:
        A flipped imag eof shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    image = image.clone()
    image = image.flip(-1)
    return image

    
def flip_vertical(image: torch.Tensor) -> torch.Tensor:
    """Flip an image vertically.
    
    Args:
        image: An image of shape [..., C, H, W].
        
    Returns:
        A flipped image of shape [..., C, H, W].
    """
    assert isinstance(image, torch.Tensor)
    image = image.clone()
    image = image.flip(-2)
    return image


def flip_image_bbox_horizontal(
    image: torch.Tensor,
    bbox : torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip an image and a bounding bbox horizontally.
    
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        
    Returns:
        A flipped image of shape [..., C, H, W].
        A flipped bbox of shape [N, 4].
    """
    center = util.get_image_center4(image)
    image  = flip_horizontal(image=image)
    bbox   = geometry.flip_bbox_horizontal(bbox=bbox, image_center=center)
    return image, bbox


def flip_image_bbox_vertical(
    image: torch.Tensor,
    bbox : torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip image and bounding bbox vertically.
    
    Args:
        image: An image of shape [..., C, H, W].
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        
    Returns:
        A flipped image of shape [..., C, H, W].
        A flipped bbox of shape [N, 4].
    """
    center = util.get_image_center4(image)
    image  = flip_vertical(image=image)
    bbox   = geometry.flip_bbox_vertical(bbox=bbox, image_center=center)
    return image, bbox
