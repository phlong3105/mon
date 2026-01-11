#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box debugging utilities.

This module provides debugging utilities for bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "draw",
]

import cv2
import numpy as np


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================


# endregion


# ==============================================================================
# region VISUALIZATION
# ==============================================================================

def draw(
    image    : np.ndarray,
    bbox     : np.ndarray,
    label    : int | str | None     = None,
    color    : tuple[int, int, int] = (255, 255, 255),
    thickness: int                  = 1,
    fill     : bool | float         = False,
    **kwargs
) -> np.ndarray:
    """Draw a bounding box on an image.

    Args:
        image: Drawing canvas, formatted as a numpy.ndarray of shape (H, W, C)
            and pixel values ranging from 0 to 255.
        bbox: Bounding box, formatted as a numpy.ndarray of shape (4+) and in
            XYXY format.
        label: Label text for the bounding box. Defaults to None.
        color: Box color as (R, G, B) values. Defaults to (255, 255, 255).
        thickness: Border thickness in pixels. Defaults to 1.
        fill: Fill transparency. If ``fill`` is True, it defaults to 0.5. If it
            is a float, it should be between 0.0 and 1.0. Defaults to False.
        kwargs: Additional keyword arguments for text rendering:
            - font_face: Font face. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            - font_scale: Font scale. Defaults to 0.5.
            - line_type: Line type. Defaults to cv2.LINE_AA.
            - shift: Number of fractional bits in the point coordinates.
                Defaults to 0.

    Returns:
        Image with drawn bounding box.
    """
    drawing = image.copy()
    h, w    = drawing.shape[:2]
    x1, y1, x2, y2 = map(int, bbox[:4])
    
    # Ensure coordinates are within bounds and ordered
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1     = max(0, min(x1, w))
    y1     = max(0, min(y1, h))
    x2     = max(0, min(x2, w))
    y2     = max(0, min(y2, h))
    
    # Handle Translucent Fill
    if fill:
        alpha = 0.5 if fill is True else float(fill)
        if x2 > x1 and y2 > y1:
            roi = drawing[y1:y2, x1:x2]
            # Create a colored rectangle of the same size as ROI
            color_block = np.full_like(roi, color, dtype=np.uint8)
            cv2.addWeighted(roi, 1 - alpha, color_block, alpha, 0, roi)
        
    # Draw Main Border
    cv2.rectangle(drawing, (x1, y1), (x2, y2), color, thickness)
    
    # Draw Label
    if label not in [None, "None", ""]:
        font      = kwargs.get("font_face", cv2.FONT_HERSHEY_SIMPLEX)
        scale     = kwargs.get("font_scale", 0.5)
        line_type = kwargs.get("line_type", cv2.LINE_AA)
        
        text      = str(label)
        (t_w, t_h), baseline = cv2.getTextSize(text, font, scale, 1)
        
        # Adjust the label position if it goes off-top
        if y1 - t_h - 4 > 0:
            text_org = (x1, y1 - 4)
            bg_pt1   = (x1, y1 - t_h - 4)
            bg_pt2   = (x1 + t_w, y1)
        else:
            text_org = (x1, y1 + t_h + 2)
            bg_pt1   = (x1, y1)
            bg_pt2   = (x1 + t_w, y1 + t_h + 4)
        
        # Draw text background
        cv2.rectangle(drawing, bg_pt1, bg_pt2, color, -1)
        
        # Draw text (Black text for better contrast on light backgrounds)
        # Simple heuristic: if sum of RGB > 382, use black text
        txt_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)
        cv2.putText(drawing, text, text_org, font, scale, txt_color, 1, line_type)
        
    return drawing

# endregion
