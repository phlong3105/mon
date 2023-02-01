# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements plotting functions."""

from __future__ import annotations

__all__ = [
    "draw_rectangle",
    "draw_rect",
]

from typing import Sequence

import cv2
import numpy as np
import torch

from mon.coreimage.typing import Ints


# region Draw Geometry Shapes

def draw_rectangle(
    image    : torch.Tensor | np.ndarray,
    rect     : torch.Tensor | np.ndarray | Sequence[int | float],
    color    : Ints,
    thickness: int = 1,
) -> torch.Tensor | np.ndarray:
    """Draw a rectangle on an image.
    
    Args:
        image: The image on which rectangle is to be drawn.
        rect: The coordinates of rectangle in [x1, y1, x2, y2] format.
        color: The color of the borderline of rectangle to be drawn in the BGR
            format.
        thickness: The thickness of the rectangle borderline in px. Thickness of
            -1 px will fill the rectangle shape by the specified color. Defaults
            to 1.
            
    Returns:
        An image.
    """
    assert len(rect)  == 4
    
    color = [color] if not isinstance(color, list) else color
    assert len(color) >= 3

    assert isinstance(image, torch.Tensor | np.ndarray)
    if isinstance(image, torch.Tensor):
        raise TypeError("This function hasn't been implemented.")
    elif isinstance(image, np.ndarray):
        image = cv2.rectangle(
            image       = image,
            start_point = (rect[0], rect[1]),
            end_point   = (rect[2], rect[3]),
            color       = color,
            thickness   = thickness,
        )
    else:
        raise TypeError(
            f":param:`image` must be a :class:`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
    return image

    
draw_rect = draw_rectangle

# endregion


# region Draw Label

def draw_bbox(

):
    pass

# endregion


'''
def draw_bbox(
    image: Tensor,
    bbox  : Tensor,
    color: Tensor | Ints = (255, 255, 255),
    fill : bool          = False
) -> Tensor:
    """Draws :param:`bbox` on :param:`image`."""
    return draw_rectangle(
        image     = image,
        rectangle = bbox,
        color     = color,
        fill      = fill,
    )
    
    
def draw_pixel(
    image: Tensor,
    x    : int,
    y    : int,
    color: Tensor | Ints = (255, 255, 255)
):
    """Draws a pixel on :param:`image`.
    
    Args:
        image: The input image to where to draw the lines with shape [C, H, W].
        x: The x coordinate of the pixel.
        y: The y coordinate of the pixel.
        color: The color of the pixel with [C] where C is the number of channels
            of the image.
    """
    image_c = util.get_num_channels(image=image)
    image[:, y, x] = color


def draw_rectangle(
    image    : Tensor,
    rectangle: Tensor,
    color    : Tensor | Ints = (255, 255, 255),
    fill     : bool          = False
) -> Tensor:
    """
    Draw N rectangles on a batch of image tensors.
    
    Args:
        image (Tensor): Tensor of [B, C, H, W].
        rectangle (Tensor): Represents number of rectangles to draw in [B, N, 4]
            N is the number of boxes to draw per batch index[x1, y1, x2, y2]
            4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
        color (Tensor | Ints | None): A size 1, size 3, [B, N, 1], or [B, N, 3]
            tensor. If C is 3, and color is 1 channel it will be broadcasted.
        fill (bool): A flag used to fill the boxes with color if True.
    
    Returns:
        This operation modifies image inplace but also returns the drawn tensor
        for convenience with same shape the of the input [B, C, H, W].
    
    Example:
        >>> img  = torch.rand(2, 3, 10, 12)
        >>> rect = torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]])
        >>> out  = draw_rectangle(img, rect)
    """
    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    if batch != batch_rect:
        raise ValueError(f"Image batch and rectangle batch must be equal.")
    if num_points != 4:
        raise ValueError(f"Number of points in rectangle must be 4.")
    
    # Clone rectangle, in case it's been expanded assignment from clipping
    # causes problems
    rectangle = rectangle.long().clone()
    
    # Clip rectangle to hxw bounds
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], 0, h - 1)
    rectangle[:, :,  ::2] = torch.clamp(rectangle[:, :,  ::2], 0, w - 1)
    
    if color is None:
        color = torch.tensor([255.0] * c).expand(batch, num_rectangle, c)
    if isinstance(color, list):
        color = torch.tensor(color)
    if len(color.shape) == 1:
        color = color.expand(batch, num_rectangle, c)
    b, n, color_channels = color.shape
    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)

    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1),
                ] = color[b, n, :, None, None]
            else:
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    rectangle[b, n, 0]
                ] = color[b, n, :, None]
                image[
                    b, :,
                    int(rectangle[b, n, 1]) : int(rectangle[b, n, 3] + 1),
                    rectangle[b, n, 2]
                ] = color[b, n, :, None]
                image[
                    b, :,
                    rectangle[b, n, 1],
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)
                ] = color[b, n, :, None]
                image[
                    b, :,
                    rectangle[b, n, 3],
                    int(rectangle[b, n, 0]) : int(rectangle[b, n, 2] + 1)
                ] = color[b, n, :, None]

    return image
'''
