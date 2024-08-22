#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Drawing.

This module implements drawing functionalities for images.
"""

from __future__ import annotations

__all__ = [
    "draw_bbox",
    "draw_heatmap",
    "draw_trajectory",
]

import cv2
import numpy as np

from mon.core.image import utils


def draw_bbox(
    image     : np.ndarray,
    bbox      : np.ndarray | list,
    label     : int | str    = None,
    color     : list[int]    = [255, 255, 255],
    thickness : int          = 1,
    line_type : int          = cv2.LINE_8,
    shift     : int          = 0,
    font_face : int          = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: int          = 0.8,
    fill      : bool | float = False,
) -> np.ndarray:
    """Draw a bounding box on an image.
    
    Args:
        image: An image.
        bbox: A bounding box in XYXY format.
        label: A label for the bounding box.
        color: A color of the bounding box.
        thickness: The thickness of the rectangle borderline in px. Thickness
            of ``-1 px`` will fill the rectangle shape by the specified color.
            Default: ``1``.
        line_type: The type of the line. One of:
            - ``cv2.LINE_4``  - 4-connected line.
            - ``cv2.LINE_8``  - 8-connected line (default).
            - ``cv2.LINE_AA`` - antialiased line.
            Default: ``cv2.LINE_8``.
        font_face: The font of the label's text. Default: ``cv2.FONT_HERSHEY_DUPLEX``.
        font_scale: The scale of the label's text. Default: ``0.8``.
        shift: The number of fractional bits in the point coordinates.
            Default: ``0``.
        fill: Fill the region inside the bounding box with transparent color.
            A float value ``[0.0-1.0]`` indicates the transparency ratio.
            A ``True`` value means ``0.5``. A value of ``1.0`` equals to
            :obj:`thickness`=-1. Default: ``False``.
    """
    drawing = image.copy()
    color   = color or [255, 255, 255]
    white   = [255, 255, 255]
    pt1     = (int(bbox[0]), int(bbox[1]))
    pt2     = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(drawing, pt1, pt2, color, thickness, line_type, shift)
    if label not in [None, "None", ""]:
        label  = f"{label}"
        offset = int(thickness / 2)
        text_size, baseline = cv2.getTextSize(label, font_face, font_scale, 1)
        cv2.rectangle(
            img       = image,
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED,
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(image, label, text_org, font_face, font_scale, white, 1)
    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = image.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)
    return drawing


def draw_heatmap(
    image     : np.ndarray,
    mask      : np.ndarray,
    color_map : int   = cv2.COLORMAP_JET,
    alpha     : float = 0.5,
    use_rgb   : bool  = False,
) -> np.ndarray:
    """Overlay a mask on the image as a heatmap. By default, the heatmap is in
    BGR format.
    
    Args:
        image: An image in RGB or BGR format.
        mask: A heatmap mask.
        color_map: A color map for the heatmap. Default: ``cv2.COLORMAP_JET``.
        alpha: The transparency ratio of the image. The final result is:
            `alpha * image + (1 - alpha) * mask`. Default: ``0.5``.
        use_rgb: If ``True``, convert the heatmap to RGB format.
            Default: ``False``.
    
    Returns:
        An image with the heatmap overlay.
    """
    if np.max(image) > 1:
        raise ValueError(f"`image` should be an `np.float32` in the range "
                         f"``[0.0, 1.0]``, but got {np.max(image)}.")
    if not 0 <= alpha <= 1:
        raise ValueError(f"`alpha` should be in the range ``[0.0, 1.0]``, "
                         f"but got: {alpha}.")
    
    if utils.is_normalized_image(mask):
        mask = np.uint8(255 * mask)
    heatmap  = cv2.applyColorMap(np.uint8(255 * mask), color_map)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap  = np.float32(heatmap) / 255
    
    drawing = (1 - alpha) * heatmap + alpha * image
    drawing = drawing / np.max(drawing)
    drawing = np.uint8(255 * drawing)
    return drawing


def draw_trajectory(
    image     : np.ndarray,
    trajectory: np.ndarray | list,
    color     : list[int] = [255, 255, 255],
    thickness : int       = 1,
    line_type : int       = cv2.LINE_8,
    point     : bool      = False,
    radius    : int       = 3,
) -> np.ndarray:
    """Draw a trajectory path on an image.
    
    Args:
        image: An image.
        trajectory: A 2D array or list of points in ``[(x1, y1), ...]`` format.
        color: A color of the bounding box.
        thickness: The thickness of the path in px. Default: 1.
        line_type: The type of the line. One of:
            - ``'cv2.LINE_4'``  - 4-connected line.
            - ``'cv2.LINE_8'``  - 8-connected line (default).
            - ``'cv2.LINE_AA'`` - antialiased line.
            Default:``' cv2.LINE_8'``.
        point: If ``True``, draw each point along the trajectory path.
            Default: ``False``.
        radius: The radius value of the point. Default: ``3``.
    """
    drawing = image.copy()
    
    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError(f"`trajectory` must be a list of points in "
                            f"``[(x1, y1), ...]`` format.")
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(drawing, [trajectory], False, color, thickness, line_type)
    if point:
        for p in trajectory:
            cv2.circle(drawing, p[0], radius, -1, color)
    return drawing
