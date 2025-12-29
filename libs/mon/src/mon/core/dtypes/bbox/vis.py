#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box visualization operations.

This module provides functions to visualize the bounding boxes for debugging
and human interaction.
"""

__all__ = [
    "draw",
]

import cv2
import numpy as np


# ==============================================================================
# CANVAS CONFIGURATION (Canvas Setup)
# ==============================================================================

# --- Style (Colormaps, Palettes, Themes) ---


# ==============================================================================
# RENDERING ENGINES (Drawing logic)
# ==============================================================================

# --- Decorate (Drawing Overlays, BBoxes, Text) ---


# --- Compose (Creating Grids and Collages) ---


# ==============================================================================
# DISPLAY & PLOTTING (High-level wrappers)
# ==============================================================================

# --- Render (Notebook/GUI display logic) ---
def draw(
    image     : np.ndarray,
    bbox      : np.ndarray,
    label     : int | str    = None,
    color     : tuple[int, int, int] = (255, 255, 255),
    thickness : int          = 1,
    line_type : int          = cv2.LINE_8,
    shift     : int          = 0,
    font_face : int          = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float        = 0.8,
    fill      : bool | float = False
) -> np.ndarray:
    """Draw a bounding box on an image.

    Args:
        image: Drawing image as a numpy.ndarray of shape (H, W, C) in range [0, 255].
        bbox: A bounding box of shape (4+) in XYXY format.
        label: Label text for the bounding box. Defaults to None.
        color: Box color as (R, G, B) values. Defaults to (255, 255, 255).
        thickness: Border thickness in pixels. Defaults to 1.
        line_type: OpenCV line type. Defaults to cv2.LINE_8.
        shift: Fractional bits in coordinates. Defaults to 0.
        font_face: OpenCV label font. Defaults to cv2.FONT_HERSHEY_DUPLEX.
        font_scale: Label text scale. Defaults to 0.8.
        fill: Fill transparency (If ``True``=0.5, 0.0-1.0). Defaults to False.

    Returns:
        An image with drawn bounding box.
    """
    drawing = image.copy()
    white   = [255, 255, 255]
    color   = color or white
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


# --- Snapshot (Saving visual previews for QA) ---
