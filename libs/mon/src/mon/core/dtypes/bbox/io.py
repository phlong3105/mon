#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box I/O operations.

This module provides input and output operations for bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "load",
]

import box
import numpy as np

from mon.core.console import error_console
from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from .ops import convert


# ==============================================================================
# region DISCOVERY
# ==============================================================================


# endregion


# ==============================================================================
# region CONNECTION
# ==============================================================================


# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def _load_coco_label(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load COCO-format labels from a JSON file.

    Args:
        path: Path to the COCO .json label file.
        remap: Mapping to remap class IDs. Defaults to None.
        verbose: Verbosity mode. Defaults to True.

    Returns:
        Batch of bounding boxes, formatted as a numpy.ndarray of shape (N, 7+)
        and in XYWH format.

    Raises:
        NotImplementedError: This method is not yet supported.
    """
    raise NotImplementedError("This method is not yet supported.")


def _load_voc_label(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load VOC-format labels from an XML file.

    Args:
        path: Path to the VOC .xml label file.
        remap: Mapping to remap class IDs. Defaults to None.
        verbose: Verbosity mode. Defaults to True.

    Returns:
        Batch of bounding boxes, formatted as a numpy.ndarray of shape (N, 7+)
        and in XYXY format.

    Raises:
        NotImplementedError: This method is not yet supported.
    """
    raise NotImplementedError("This method is not yet supported.")


def _load_yolo_label(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load YOLO-format labels from a text file.

    Each line in the file should contain:
    <class_id> <center_x> <center_y> <width> <height> <angle> | Optional: <confidence>
    where:
        - <class_id> is the class index (0-based).
        - <x_center>, <y_center>, <width>, and <height> are normalized values
          relative to the image dimensions.
        - <angle> is the rotation angle in degrees. For horizontal bounding
          boxes, this value is typically 0.
        - <confidence> is an optional value representing the confidence score.

    Args:
        path: Path to the YOLO .txt label file.
        remap: Mapping to remap class IDs. Defaults to None.
        verbose: Verbosity mode. Defaults to True.

    Returns:
        Batch of bounding boxes, formatted as a numpy.ndarray of shape (N, 7+)
        and in CXCYWHN format.
    """
    path = Path(path).normalize()
    if not path.is_txt_file(exist=True):
        if verbose:
            error_console.print(f"Path must be a valid .txt file: {path}")
        return np.empty((0, 7), dtype=np.float32)

    try:
        # Using np.loadtxt is significantly faster for large label files
        # It handles whitespace stripping and conversion in one pass
        raw_data = np.loadtxt(path, dtype=np.float32, ndmin=2)
    except Exception as e:
        if verbose:
            error_console.print(f"Failed to parse {path}: {e}")
        return np.empty((0, 7), dtype=np.float32)

    # Standardize Columns
    # YOLO Standard: [class, cx, cy, w, h]        -> 5 columns
    # YOLO OBB:      [class, cx, cy, w, h, angle] -> 6 columns
    # YOLO + Conf:   [class, cx, cy, w, h, conf]  -> 6 columns
    if raw_data.size == 0:
        return np.empty((0, 7), dtype=np.float32)

    num_cols = raw_data.shape[1]

    # If file only has [cx, cy, w, h], prepend a dummy class 0
    if num_cols == 4:
        raw_data = np.column_stack([np.zeros(len(raw_data)), raw_data])
        num_cols = 5

    # Class Remapping
    if remap:
        # Efficiently remap all class IDs at once using np.vectorize or mapping
        raw_data[:, 0] = np.array([remap.get(int(c), c) for c in raw_data[:, 0]])

    # Format Normalization (Building the N x 7+ matrix)
    # Output Format: [cx_n, cy_n, w_n, h_n, angle, conf, class, ...id]
    # Note: The docstring says CXCYWHN which usually implies [cx, cy, w, h, ...]
    # The previous implementation had:
    # final_bboxes[:, 5] = cls
    # But standard mon/yolo usually expects: [cx, cy, w, h, conf, class] or similar.
    # Let's check the BBox class definition in core.py (from previous turn).
    # BBox attributes:
    # conf -> index 5
    # cls -> index 6
    # id -> index 7
    # So the target layout must be: [cx, cy, w, h, angle, conf, class, id]

    n_rows       = raw_data.shape[0]
    # We need at least 7 columns for [cx, cy, w, h, angle, conf, class]
    # Initialize with zeros
    final_bboxes = np.zeros((n_rows, 7), dtype=np.float32)

    cls  = raw_data[:, 0]
    cxcy = raw_data[:, 1:3]
    wh   = raw_data[:, 3:5]

    final_bboxes[:, 0:2] = cxcy  # cx, cy
    final_bboxes[:, 2:4] = wh    # w, h
    final_bboxes[:, 6]   = cls   # class is at index 6 based on BBox class

    # Angle vs. Confidence Ambiguity Logic
    if num_cols == 6:
        col5 = raw_data[:, 5]
        # Heuristic: YOLO confidence is usually 0.0-1.0.
        # Angles are usually > 1.0 or 0.0.
        if np.all((col5 >= 0) & (col5 <= 1.0)) and not np.any(col5 > 0.999):
            # Likely confidence
            final_bboxes[:, 4] = 0.0   # angle
            final_bboxes[:, 5] = col5  # conf
        else:
            # Likely angle
            final_bboxes[:, 4] = col5  # angle
            final_bboxes[:, 5] = 1.0   # default conf
    elif num_cols >= 7:
        final_bboxes[:, 4] = raw_data[:, 5]  # angle
        # If we have more columns, we might have conf and id
        # But standard YOLO usually is [class, cx, cy, w, h, conf] or [class, cx, cy, w, h, angle, conf]
        # The raw_data here is [class, cx, cy, w, h, angle, conf]
        # So raw_data[:, 6] would be conf
        if num_cols > 6:
             final_bboxes[:, 5] = raw_data[:, 6] # conf

        # If there are even more columns, append them (e.g. track id)
        if num_cols > 7:
             final_bboxes = np.column_stack([final_bboxes, raw_data[:, 7:]])
    else:
        # Default confidence
        final_bboxes[:, 5] = 1.0

    return final_bboxes


def load(
    path   : Path,
    fmt    : BBoxFormat,
    imgsz  : tuple[int, int],
    remap  : dict | box.Box = None,
    verbose: bool = False
) -> np.ndarray:
    """Load bounding boxes from a label file.

    Load bounding boxes from a label file and optionally convert to the desired
    format.

    Args:
        path: Label file path (YOLO .txt, VOC .xml, COCO .json).
        fmt: Desired target format or conversion code.
        imgsz: Image size as (H, W) required for format conversions.
        remap: Mapping for class IDs or names. Defaults to None.
        verbose: Verbosity mode. Defaults to False.

    Returns:
        Batch of bounding boxes, formatted as a numpy.ndarray of shape (N, 7+)
        and in the desired format.

    Raises:
        ValueError: If ``path`` is invalid or if conversion parameters are missing.
    """
    path = Path(path).normalize()  # Ensure absolute, clean path

    # Determine Source vs Target Formats
    # Use BBoxFormat internal logic to distinguish between a static format
    # and a conversion instruction (e.g., 'voc_to_yolo')
    fmt = BBoxFormat(value=fmt)

    if fmt in BBoxFormat.conversion_codes():
        # Example: 'coco_to_cxcywhn' -> src_fmt = 'coco', target_fmt = 'cxcywhn'
        src_fmt_str, target_fmt_str = fmt.value.split("_to_")
        src_fmt    = BBoxFormat(src_fmt_str)
        target_fmt = BBoxFormat(target_fmt_str)
    else:
        # No conversion requested, just load in original format
        src_fmt    = fmt
        target_fmt = None

    # Match Reader based on source format
    # We use a mapping dictionary for cleaner expansion later
    readers = {
        BBoxFormat.COCO:    _load_coco_label,
        BBoxFormat.XYWH:    _load_coco_label,
        BBoxFormat.VOC:     _load_voc_label,
        BBoxFormat.XYXY:    _load_voc_label,
        BBoxFormat.YOLO:    _load_yolo_label,
        BBoxFormat.CXCYWHN: _load_yolo_label,
    }

    if src_fmt not in readers:
        raise ValueError(
            f"Unsupported bounding box format: {src_fmt.value}. "
            f"Must be one of: {list(readers.keys())}."
        )

    # Execute the read operation
    bbox = readers[src_fmt](path, remap=remap, verbose=verbose)

    # Handle Conditional Conversion
    if target_fmt:
        if imgsz is None:
            raise ValueError(
                f"Expected 'imgsz' for conversion from {src_fmt.value} to {target_fmt.value}, "
                f"but got None."
            )

        # Build the specific conversion instruction for the convert() utility
        conversion_code = BBoxFormat(f"{src_fmt.value}_to_{target_fmt.value}")
        bbox            = convert(bbox=bbox, fmt=conversion_code, imgsz=imgsz)

    return bbox

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================


# endregion
