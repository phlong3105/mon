#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instance annotation visualization operations.

This module provides functions to visualize the instance annotations for
debugging and human interaction.
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
    trajectory: np.ndarray,
    color     : tuple[int, int, int] = (255, 255, 255),
    thickness : int  = 1,
    line_type : int  = cv2.LINE_8,
    point     : bool = False,
    radius    : int  = 3
) -> np.ndarray:
    """Draw a trajectory path on an image.

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
            raise TypeError("``trajectory`` must be a list of points in [(x1, y1), ...] format.")
        trajectory = np.array(trajectory)
    
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(drawing, [trajectory], False, color, thickness, line_type)
    
    if point:
        for p in trajectory:
            cv2.circle(drawing, tuple(p[0]), radius, color, -1)  # Fixed syntax and type

    return drawing


# --- Snapshot (Saving visual previews for QA) ---
