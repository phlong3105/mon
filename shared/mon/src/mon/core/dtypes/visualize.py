#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements visualization functions."""

__all__ = [
    "draw_bbox",
    "draw_trajectory",
]

from typing import Union

import cv2
import numpy as np


def draw_bbox(
    image     : np.ndarray,
    bbox      : Union[np.ndarray, list],
    label     : int | str    = None,
    color     : tuple[int, int, int] = (255, 255, 255),
    thickness : int          = 1,
    line_type : int          = cv2.LINE_8,
    shift     : int          = 0,
    font_face : int          = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float        = 0.8,
    fill      : bool | float = False
) -> np.ndarray:
    """Draw bounding box on an image.

    Args:
        image: Drawing image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`
            in range :math:`[0, 255]`.
        bbox: Bounding boxes as a ``numpy.ndarray`` or ``list`` in ``XYXY`` format.
        label: Label for box. Default: ``None``.
        color: Box color as ``tuple`` of :math:`(R, G, B)` values.
            Default: ``(255, 255, 255)``.
        thickness: Border thickness in px as ``int``. Default: ``1``.
        line_type: OpenCV line type. Default: ``cv2.LINE_8``.
        shift: Fractional bits in coordinates. Default: ``0``.
        font_face: OpenCV label font. Default: ``cv2.FONT_HERSHEY_DUPLEX``.
        font_scale: Label text scale. Default: ``0.8``
        fill: Fill transparency (If ``True``=0.5, 0.0-1.0). Default: ``False``.

    Returns:
        An image with drawn bounding box.
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
            img       = drawing,  # Changed from 'image' to 'drawing' for consistency
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(drawing, label, text_org, font_face, font_scale, white, 1)

    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = drawing.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)

    return drawing


def draw_trajectory(
    image     : np.ndarray,
    trajectory: Union[np.ndarray, list],
    color     : tuple[int, int, int] = (255, 255, 255),
    thickness : int  = 1,
    line_type : int  = cv2.LINE_8,
    point     : bool = False,
    radius    : int  = 3
) -> np.ndarray:
    """Draw trajectory path on an image.

    Args:
        image: Drawing image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`
            in range :math:`[0, 255]`.
        trajectory: 2D points as a ``numpy.ndarray`` or ``list`` of shape :math:`[(x1, y1), ...]`.
        color: Path color as ``tuple`` of :math:`(R, G, B)` values.
            Default: ``(255, 255, 255)``.
        thickness: Path thickness in px. Default: ``1``.
        line_type: OpenCV line type. Default: ``cv2.LINE_8``.
        point: Draw points if ``True``. Default: ``False``.
        radius: Point radius in px. Default: ``3``.

    Returns:
        An image with drawn trajectories.

    Raises:
        TypeError: If ``trajectory`` format is invalid.
    """
    drawing = image.copy()

    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError("[trajectory] must be a list of points in [(x1, y1), ...] format.")
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(drawing, [trajectory], False, color, thickness, line_type)
    if point:
        for p in trajectory:
            cv2.circle(drawing, tuple(p[0]), radius, color, -1)  # Fixed syntax and type

    return drawing
