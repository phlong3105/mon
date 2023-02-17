#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements drawing functions."""

from __future__ import annotations

__all__ = [
    "draw_bbox",
    "draw_contour",
    "draw_segment",
    "draw_trajectory",
]

import cv2
import numpy as np


# region Geometry

def draw_bbox(
    image     : np.ndarray,
    bbox      : np.ndarray | list,
    label     : int | str | None = None,
    color     : list[int] | None = None,
    thickness : int              = 1,
    line_type : int              = cv2.LINE_8,
    shift     : int              = 0,
    font_face : int              = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: int              = 0.8,
    fill      : bool | float     = False,
) -> np.ndarray:
    """Draw a bounding box on an image.
    
    Args:
        image: An image.
        bbox: A bounding box in XYXY format.
        label: A label or ID of the object inside the bounding box.
        color: A color of the bounding box.
        thickness: The thickness of the rectangle border line in px. Thickness
            of -1 px will fill the rectangle shape by the specified color.
            Defaults to 1.
        line_type: The type of the line. One of:
            - cv2.LINE_4 - 4-connected line.
            - cv2.LINE_8 - 8-connected line (default).
            - cv2.LINE_AA - antialiased line.
            Defaults to cv2.LINE_8.
        font_face: The font of the label's text. Defaults to
            cv2.FONT_HERSHEY_DUPLEX.
        font_scale: The scale of the label's text. Defaults to 0.8.
        shift: The number of fractional bits in the point coordinates. Defaults
            to 0.
        fill: Fill the region inside the bounding box with transparent color. A
            float value [0.0-1.0] indicates the transparency ratio. A True value
            means 0.5. A value of 1.0 equals to :param:`thickness` = -1.
            Defaults to False.
    """
    drawing = image.copy()
    color   = color or [255, 255, 255]
    pt1     = (int(bbox[0]), int(bbox[1]))
    pt2     = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(
        img       = drawing,
        pt1       = pt1,
        pt2       = pt2,
        color     = color,
        thickness = thickness,
        lineType  = line_type,
        shift     = shift,
    )
    if label is not None:
        label  = f"{label}"
        offset = int(thickness / 2)
        text_size, baseline = cv2.getTextSize(
            text      = label,
            fontFace  = font_face,
            fontScale = font_scale,
            thickness = 1
        )
        cv2.rectangle(
            img       = image,
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED,
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(
            img       = image,
            text      = label,
            org       = text_org,
            fontFace  = font_face,
            fontScale = font_scale,
            color     = [255, 255, 255],
            thickness = 1
        )
    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = image.copy()
        cv2.rectangle(
            img       = overlay,
            pt1       = pt1,
            pt2       = pt2,
            color     = color,
            thickness = -1,
        )
        cv2.addWeighted(
            src1  = overlay,
            alpha = alpha,
            src2  = drawing,
            beta  = 1 - alpha,
            gamma = 0,
            dst   = drawing,
        )
    return drawing


def draw_segment(
    image     : np.ndarray,
    segment   : np.ndarray | list,
    bbox      : np.ndarray | bool | None = None,
    label     : int | str | None = None,
    color     : list[int] | None = None,
    thickness : int              = 1,
    line_type : int              = cv2.LINE_8,
    shift     : int              = 0,
    font_face : int              = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: int              = 0.8,
    fill      : bool | float     = False,
    point     : bool             = False,
    radius    : int              = 3,
) -> np.ndarray:
    """Draw a segmentation mask on an image.
    
    Args:
        image: An image.
        segment: A segmentation mask in [(x1, y1), ...] format.
        bbox: A bounding box of the segmentation. If True, generate the bounding
            box using :meth:`cv2.boundingRect()`. If False or None, then skip.
        label: A label or ID of the object.
        color: A color of the bounding box.
        thickness: The thickness of the segmentation mask border line in px.
            Thickness of -1 px will fill the rectangle shape by the specified
            color. Defaults to 1.
        line_type: The type of the line. One of:
            - cv2.LINE_4 - 4-connected line.
            - cv2.LINE_8 - 8-connected line (default).
            - cv2.LINE_AA - antialiased line.
            Defaults to cv2.LINE_8.
        font_face: The font of the label's text. Defaults to
            cv2.FONT_HERSHEY_DUPLEX.
        font_scale: The scale of the label's text. Defaults to 0.8.
        shift: The number of fractional bits in the point coordinates. Defaults
            to 0.
        fill: Fill the region inside the segment with transparent color. A float
            value [0.0-1.0] indicates the transparency ratio. A True value means
            0.5. Defaults to False.
        point: If True, draw each point along the segment contour. Defaults to
            False.
        radius: The radius value of the point. Defaults to 3.
    """
    drawing = image.copy()
    
    if isinstance(segment, list):
        if not all(len(t) == 2 for t in segment):
            raise TypeError(
                f"trajectory must be a list of points in [(x1, y1), ...] format."
            )
        segment = np.array(segment)
    segment = np.array(segment).reshape((-1, 1, 2)).astype(int)

    if bbox is True or (bbox in [False, None] and label is not None):
        x, y, w, h = cv2.boundingRect(segment)
        bbox       = [x, y, x + w, y + h]
        
    color = color or [255, 255, 255]
    """
    cv2.polylines(
        img       = image,
        pts       = [segment],
        isClosed  = True,
        color     = color,
        thickness = thickness,
        lineType  = line_type,
    )
    """
    cv2.drawContours(
        image      = drawing,
        contours   = [segment],
        contourIdx = -1,
        color      = color,
        thickness  = thickness,
        lineType   = line_type,
        hierarchy  = None,
        maxLevel   = None,
        offset     = None,
    )
    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = drawing.copy()
        cv2.fillPoly(
            img   = overlay,
            pts   = [segment],
            color = color,
        )
        cv2.addWeighted(
            src1  = overlay,
            alpha = alpha,
            src2  = drawing,
            beta  = 1 - alpha,
            gamma = 0,
            dst   = drawing,
        )
    if bbox not in [False, None]:
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(
            img       = drawing,
            pt1       = pt1,
            pt2       = pt2,
            color     = color,
            thickness = thickness if thickness > -1 else 1,
            lineType  = line_type,
            shift     = shift,
        )
        if label is not None:
            label  = f"{label}"
            offset = int(thickness / 2)
            text_size, baseline = cv2.getTextSize(
                text      = label,
                fontFace  = font_face,
                fontScale = font_scale,
                thickness = 1
            )
            cv2.rectangle(
                img       = drawing,
                pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
                pt2       = (pt1[0] + text_size[0], pt1[1]),
                color     = color,
                thickness = cv2.FILLED,
            )
            text_org = (pt1[0] - offset, pt1[1] - offset)
            cv2.putText(
                img       = drawing,
                text      = label,
                org       = text_org,
                fontFace  = font_face,
                fontScale = font_scale,
                color     = [255, 255, 255],
                thickness = 1
            )
    if point:
        for p in segment:
            cv2.circle(
                img       = drawing,
                center    = p[0],
                radius    = radius,
                thickness = -1,
                color     = color
            )
    return drawing


draw_contour = draw_segment


def draw_trajectory(
    image     : np.ndarray,
    trajectory: np.ndarray | list,
    color     : list[int] | None = None,
    thickness : int              = 1,
    line_type : int              = cv2.LINE_8,
    point     : bool             = False,
    radius    : int              = 3,
) -> np.ndarray:
    """Draw a trajectory path on an image.
    
    Args:
        image: An image.
        trajectory: A 2-D array or list of points in [(x1, y1), ...] format.
        color: A color of the bounding box.
        thickness: The thickness of the path in px. Defaults to 1.
        line_type: The type of the line. One of:
            - cv2.LINE_4 - 4-connected line.
            - cv2.LINE_8 - 8-connected line (default).
            - cv2.LINE_AA - antialiased line.
            Defaults to cv2.LINE_8.
        point: If True, draw each point along the trajectory path. Defaults to
            False.
        radius: The radius value of the point. Defaults to 3.
    """
    drawing = image.copy()
    
    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError(
                f"trajectory must be a list of points in [(x1, y1), ...] format."
            )
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    
    color = color or [255, 255, 255]
    cv2.polylines(
        img       = drawing,
        pts       = [trajectory],
        isClosed  = False,
        color     = color,
        thickness = thickness,
        lineType  = line_type,
    )
    if point:
        for p in trajectory:
            cv2.circle(
                img       = drawing,
                center    = p[0],
                radius    = radius,
                thickness = -1,
                color     = color
            )
    return drawing

# endregion
