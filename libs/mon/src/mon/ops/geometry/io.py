#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label I/O Operation.

This module provides input and output operations for label files.
"""

from __future__ import annotations

__all__ = [
    "load_bbox",
    "load_bbox_yolo",
]

import numpy as np
from box import Box
from numpy import ndarray

from mon.core import BBoxes, BBoxFormat, load_yaml, Path, Size
from mon.ops.image import read_imgsz

# ==============================================================================
# region CONSTANTS
# ==============================================================================

_DEFAULT_ANGLE = 0.0
_DEFAULT_SCORE = 0.0
_DEFAULT_TRACK_ID = -1.0

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def load_bbox_yolo(
    path: Path,
    remap: dict | Path | None = None,
    has_angle: bool = False,
    has_score: bool = False,
    has_track_id: bool = False,
    *args, **kwargs
) -> ndarray:
    """Load all bounding boxes in a YOLO-format .txt file.

    Each line in the label file should contain:
    class_id, cx, cy, w, h, angle, score, track_id
    where:
        - class_id: is the class index (0-based).
        - cx, cy, w, h: are normalized values relative to the image dimensions.
        - angle: is the rotation angle in degrees. For horizontal bounding
          boxes, this value is typically 0.
        - score: is an optional value representing the confidence score.
        - track_id: is an optional value representing the tracking ID.

    Args:
        path (Path): Path to the YOLO .txt label file.
        remap (dict | Path, optional): Mapping to remap class IDs. Can be a
            dictionary or a path to a YAML file containing the mapping.
            Defaults to None.
        has_angle (bool): Whether the label file includes angle information.
            Defaults to False.
        has_score (bool): Whether the label file includes score information.
            Defaults to False.
        has_track_id (bool): Whether the label file includes tracking ID.
            Defaults to False.

    Returns:
        ndarray: An array of shape (N, 8+) where each row represents a bounding
            box in the format [cx, cy, w, h, angle, class_id, score, track_id].
    """
    path = Path(path).normalize()

    # 1. Handle invalid or empty label file
    if not path.is_txt_file(exist=True):
        return np.empty((0, 8), dtype=np.float32)

    # Using np.loadtxt is significantly faster for large label files. It handles
    # whitespace stripping and conversion in one pass.
    try:
        raw_data = np.loadtxt(path.as_posix(), dtype=np.float32, ndmin=2)
        if raw_data.size == 0:
            return np.empty((0, 8), dtype=np.float32)
    except Exception as e:
        return np.empty((0, 8), dtype=np.float32)

    # 2. Handle remapping class ids
    if remap:
        if isinstance(remap, (Path, str)):
            remap = load_yaml(path=remap)
        remap = getattr(remap, "remap", remap)
        if not isinstance(remap, (Box, dict)):
            raise TypeError(f"expected remap to be a dict, got {type(remap).__name__}.")

    # 3. Parse raw data
    num_rows = raw_data.shape[0]
    num_cols = raw_data.shape[1]
    # If only has [cx, cy, w, h], prepend dummy class "0" --> [0, cx, cy, w, h]
    if num_cols == 4:
        raw_data = np.column_stack([np.zeros(num_rows), raw_data])
    # Remap class IDs if remap is provided
    if remap:
        # Efficiently remap all class IDs at once using np.vectorize or mapping
        raw_data[:, 0] = np.array([remap.get(int(c), c) for c in raw_data[:, 0]])

    # 4. Construct the final bounding box array
    bbox = np.zeros((num_rows, 8), dtype=np.float32)
    bbox[:, 0:4] = raw_data[:, 1:5]  # cx, cy, w, h
    bbox[:, 5] = raw_data[:, 0]  # class_id
    if has_angle and has_score and has_track_id:
        bbox[:, 4] = raw_data[:, 5]  # angle
        bbox[:, 6] = raw_data[:, 6]  # score
        bbox[:, 7] = raw_data[:, 7]  # track_id
    elif (not has_angle) and has_score and has_track_id:
        bbox[:, 4] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = raw_data[:, 5]  # score
        bbox[:, 7] = raw_data[:, 6]  # track_id
    elif has_angle and (not has_score) and has_track_id:
        bbox[:, 4] = raw_data[:, 5]  # angle
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = raw_data[:, 6]  # track_id
    elif has_angle and has_score and (not has_track_id):
        bbox[:, 4] = raw_data[:, 5]  # angle
        bbox[:, 6] = raw_data[:, 6]  # score
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    elif (not has_angle) and (not has_score) and has_track_id:
        bbox[:, 4] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = raw_data[:, 5]  # track_id
    elif (not has_angle) and has_score and (not has_track_id):
        bbox[:, 4] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = raw_data[:, 5]  # score
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    elif has_angle and (not has_score) and (not has_track_id):
        bbox[:, 4] = raw_data[:, 5]  # angle
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    else:  # not has_angle and not has_score and not has_track_id:
        bbox[:, 4] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)

    # 5. Validate
    if bbox[:, 0:4].any() < 0:
        raise ValueError("bbox coordinates must be non-negative.")

    return bbox


def load_bbox(
    path: Path,
    fmt: BBoxFormat,
    remap: dict | Path | None = None,
    imgsz: Size | None = None,
    image_file: Path | None = None,
    as_array: bool = False,
    *args, **kwargs
) -> BBoxes | ndarray:
    """Load bounding boxes from a file.

    Args:
        path (PathLike): Path to the label file.
        fmt (BBoxFormat): Format of the bounding boxes in the file.
        remap (dict | Path, optional): Class ID remapping. Defaults to None.
        imgsz (Size, optional): Image size (width, height). Required if
            ``as_array`` is False and ``image_file`` is not provided.
            Defaults to None.
        image_file (Path, optional): Path to the corresponding image file.
            Used to infer image size if `imgsz` is not provided. Defaults to None.
        as_array (bool, optional): If True, return the bounding box array,
            otherwise return a BBoxes instance. Defaults to False.
    """
    # 1. Normalize inputs
    fmt = BBoxFormat(fmt)

    # 2. Load bounding boxes array
    if fmt == BBoxFormat.CXCYWHN:
        bbox = load_bbox_yolo(path=path, remap=remap, *args, **kwargs)
    else:
        raise ValueError(f"the loading method for '{fmt}' format has not been "
                         f"supported yet.")

    # 3. Return bbox array if requested
    if as_array:
        return bbox

    # 4. Validate inputs
    if imgsz:
        imgsz = Size.from_any(imgsz)
    elif image_file:
        image_file = Path(image_file).normalize()
        if image_file.is_image_file(exists=True):
            imgsz = read_imgsz(image_file)
    else:
        raise ValueError(f"expected either imgsz or image_file to be provided "
                         f"when 'as_array=False', "
                         f"got imgsz={imgsz} and image_file={image_file}.")

    # 5. Convert to BBoxes instance if requested
    if fmt == BBoxFormat.CXCYWHN:
        bbox = BBoxes(bbox=bbox, imgsz=imgsz, path=path)
    elif fmt == BBoxFormat.XYXY:
        bbox = BBoxes.from_xyxy(bbox=bbox, imgsz=imgsz, path=path)
    elif fmt == BBoxFormat.XYWH:
        bbox = BBoxes.from_xywh(bbox=bbox, imgsz=imgsz, path=path)
    else:
        raise ValueError(f"unsupported bbox format {fmt}, "
                         f"must be one of {BBoxFormat.formats()}.")

    return bbox

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

def write_bbox_yolo(
    bbox: BBoxes,
    path: Path,
    imgsz: Size | None = None,
    *args, **kwargs
):
    """Write bounding boxes to a YOLO-format .txt file.

    Each line in the label file should contain:
    class_id, cx, cy, w, h, angle, score, track_id
    where:
        - class_id: is the class index (0-based).
        - cx, cy, w, h: are normalized values relative to the image dimensions.
        - angle: is the rotation angle in degrees. For horizontal bounding
          boxes, this value is typically 0.
        - score: is an optional value representing the confidence score.
        - track_id: is an optional value representing the tracking ID.

    Args:
        bbox (BBoxes): The bounding boxes to be written.
        path (Path): The file path to write the bounding boxes to.
        imgsz (Size | None, optional): The image size (width, height) to use for
            normalization if needed. Required if ``fmt`` is a normalized format.
    """
    # Normalize inputs
    path = Path(path).normalize()

    # Create parent directory
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Convert bbox to the desired output format
    bbox_ = bbox.cxcywhn(imgsz=imgsz)

    # Write bboxes to label file
    with open(path.as_posix(), "w", encoding="utf-8") as f:
        for b in bbox_:
            f.write(
                f"{int(b[0])} "
                f"{float(b[1])} {float(b[2])} {float(b[3])} {float(b[4])} "
                f"{float(b[5])} "
                f"{float(b[6])} "
                f"{int(b[7])} "
                f"\n"
            )


def write_bbox(
    bbox: BBoxes,
    path: Path,
    fmt: BBoxFormat,
    imgsz: Size | None = None,
    *args, **kwargs
):
    """Write bounding boxes to a YOLO-format .txt file.

    Each line in the label file should contain:
    class_id, cx, cy, w, h, angle, score, track_id
    where:
        - class_id: is the class index (0-based).
        - cx, cy, w, h: are normalized values relative to the image dimensions.
        - angle: is the rotation angle in degrees. For horizontal bounding
          boxes, this value is typically 0.
        - score: is an optional value representing the confidence score.
        - track_id: is an optional value representing the tracking ID.

    Args:
        bbox (BBoxes): The bounding boxes to be written.
        path (Path): The file path to write the bounding boxes to.
        imgsz (Size, optional): The image size (width, height) to use for
            normalization if needed. Required if ``fmt`` is a normalized format.
    """

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
