#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label I/O Operation.

This module provides input and output operations for label files.
"""

from __future__ import annotations

__all__ = [
    "convert_labels_to_json",
    "load_bbox",
    "write_bbox",
]

import json
import xml.etree.ElementTree as ET

import numpy as np
from box import Box
from numpy import ndarray

from mon.core import (
    BBoxes,
    BBoxFormat,
    create_progress_bar,
    K,
    load_yaml,
    Path,
    Size,
)
from mon.ops.image import read_imgsz
from .bbox import to_2d_bbox

# ==============================================================================
# region CONSTANTS
# ==============================================================================

_DEFAULT_ANGLE: float = 0.0
_DEFAULT_SCORE: float = 0.0
_DEFAULT_TRACK_ID: float = -1.0

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

# endregion


# ==============================================================================
# region INPUT
# ==============================================================================

def _load_bbox_yolo(
    path: Path,
    remap: dict | Path | None = None,
    has_angle: bool = False,
    has_score: bool = False,
    has_track_id: bool = False,
    *args, **kwargs
) -> ndarray:
    """Load all bounding boxes in a YOLO-format .txt file.

    Each line in the label file should contain:
       0       1   2  3  4    5      6       7
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
            box in the format [class_id, cx, cy, w, h, angle, score, track_id].
    """
    path = Path(path).normalize()

    # 1. Handle invalid or empty label file
    if not path.has_ext(".txt", exists=True):
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
    bbox[:, 0] = raw_data[:, 0]  # class_id
    bbox[:, 1:5] = raw_data[:, 1:5]  # cx, cy, w, h
    if has_angle and has_score and has_track_id:
        bbox[:, 5] = raw_data[:, 5]  # angle
        bbox[:, 6] = raw_data[:, 6]  # score
        bbox[:, 7] = raw_data[:, 7]  # track_id
    elif (not has_angle) and has_score and has_track_id:
        bbox[:, 5] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = raw_data[:, 5]  # score
        bbox[:, 7] = raw_data[:, 6]  # track_id
    elif has_angle and (not has_score) and has_track_id:
        bbox[:, 5] = raw_data[:, 5]  # angle
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = raw_data[:, 6]  # track_id
    elif has_angle and has_score and (not has_track_id):
        bbox[:, 5] = raw_data[:, 5]  # angle
        bbox[:, 6] = raw_data[:, 6]  # score
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    elif (not has_angle) and (not has_score) and has_track_id:
        bbox[:, 5] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = raw_data[:, 5]  # track_id
    elif (not has_angle) and has_score and (not has_track_id):
        bbox[:, 5] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = raw_data[:, 5]  # score
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    elif has_angle and (not has_score) and (not has_track_id):
        bbox[:, 5] = raw_data[:, 5]  # angle
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)
    else:  # not has_angle and not has_score and not has_track_id:
        bbox[:, 5] = np.full(num_rows, _DEFAULT_ANGLE)
        bbox[:, 6] = np.full(num_rows, _DEFAULT_SCORE)
        bbox[:, 7] = np.full(num_rows, _DEFAULT_TRACK_ID)

    # 5. Validate
    if bbox[:, 1:5].any() < 0:
        raise ValueError("bbox coordinates must be non-negative.")

    return bbox


def _load_bbox_voc(
    path: Path,
    remap: dict | Path | None = None,
    *args, **kwargs
) -> ndarray:
    """Load bounding boxes from a VOC-format XML file.

    Args:
        path (Path): Path to the VOC XML label file.
        remap (dict | Path, optional): Mapping to remap class IDs. Can be a
            dictionary or a path to a YAML file containing the mapping.
            Defaults to None.

    Returns:
        ndarray: An array of shape (N, 8+) where each row represents a bounding
            box in the format [class_id, x1, y1, x2, y2, angle, score, track_id].
    """
    path = Path(path).normalize()

    # 1. Handle invalid or empty label file
    if not path.has_ext(".xml", exists=True):
        return np.empty((0, 8), dtype=np.float32)

    try:
        tree = ET.parse(path.as_posix())
        root = tree.getroot()
        if root is None:
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
    # size = root.find("size")
    # width = int(size.find("width").text)
    # height = int(size.find("height").text)
    # depth = int(size.find("height").text)

    bbox = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        class_id = remap.get(label, 0) if remap else 0

        # Extract coordinates as integers
        bndbox = obj.find("bndbox")
        x1 = int(bndbox.find("xmin").text)
        y1 = int(bndbox.find("ymin").text)
        x2 = int(bndbox.find("xmax").text)
        y2 = int(bndbox.find("ymax").text)
        bbox.append([class_id, x1, y1, x2, y2, _DEFAULT_ANGLE, _DEFAULT_SCORE, _DEFAULT_TRACK_ID])

    # 4. Convert to a numpy array
    if len(bbox) == 0:
        bbox = np.empty((0, 8), dtype=np.float32)
    else:
        bbox = np.array(bbox, dtype=np.float32)
        bbox = to_2d_bbox(bbox)

    # 5. Validate
    if bbox[:, 1:5].any() < 0:
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
    # Normalize inputs
    fmt = BBoxFormat(fmt)

    # Load bounding boxes array
    if fmt == BBoxFormat.CXCYWHN:
        bbox = _load_bbox_yolo(path=path, remap=remap, *args, **kwargs)
    elif fmt == BBoxFormat.XYXY:
        bbox = _load_bbox_voc(path=path, remap=remap, *args, **kwargs)
    else:
        raise ValueError(f"the loading method for '{fmt}' format has not been "
                         f"supported yet.")

    # Return bbox array if requested
    if as_array:
        return bbox
    # Convert to BBoxes instance if requested
    else:
        if imgsz:
            imgsz = Size.from_any(imgsz)
        elif image_file:
            image_file = Path(image_file).normalize()
            if image_file.is_image_file(exists=True):
                imgsz = read_imgsz(image_file)

        if not isinstance(imgsz, Size):
            raise ValueError(
                f"expected either imgsz or image_file to be provided when "
                f"'as_array=False', got imgsz={imgsz} and image_file={image_file}"
            )

        return BBoxes.from_any(bbox=bbox, imgsz=imgsz, fmt=fmt, path=path)

# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

def _write_bbox_yolo(
    bbox: BBoxes,
    path: Path,
    imgsz: Size | None = None,
    *args, **kwargs
):
    """Write bounding boxes to a YOLO-format .txt file.

    Each line in the label file should contain:
       0       1   2  3  4    5      6       7
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

    # Write bboxes to label file
    with open(path.as_posix(), "w", encoding="utf-8") as f:
        for i, b in enumerate(bbox):
            # [class_id, cx, cy, w, h, angle, score, track_id]
            b_ = b.cxcywhn(imgsz=imgsz)
            f.write(
                f"{b.class_id} "  # class_id
                f"{range(float(b_[0]), 30)} "  # cx
                f"{range(float(b_[1]), 30)} "  # cy
                f"{range(float(b_[2]), 30)} "  # w  
                f"{range(float(b_[3]), 30)} "  # h  
                f"{b[4]} "  # angle
                f"{b[6]} "
                f"{b[7]} "
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
       0       1   2  3  4    5      6       7
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
        fmt (BBoxFormat): The format to write the bounding boxes in.
        imgsz (Size, optional): The image size (width, height) to use for
            normalization if needed. Required if ``fmt`` is a normalized format.
    """
    # Normalize inputs
    fmt = BBoxFormat(fmt)

    # Write bounding boxes in the desired format
    if fmt == BBoxFormat.CXCYWHN:
        _write_bbox_yolo(bbox=bbox, path=path, imgsz=imgsz, *args, **kwargs)
    else:
        raise ValueError(f"the writing method for '{fmt}' format has not been "
                         f"supported yet.")


# ==============================================================================
# region INPUT-OUTPUT
# ==============================================================================

def convert_labels_to_json(
    image_dir: Path,
    label_dir: Path,
    output_json: Path,
    fmt: BBoxFormat = BBoxFormat.CXCYWHN,
    remap: dict | Path | None = None,
):
    """Convert all labels from a directory to a JSON file.

    Args:
        image_dir (Path): Directory containing the image files.
        label_dir (Path): Directory containing the label files.
        output_json (Path): Path to save the output JSON file.
        fmt (BBoxFormat, optional): Format of the bounding boxes in the label files.
            Defaults to BBoxFormat.CXCYWHN.
        remap (dict | Path, optional): Class ID remapping. Defaults to None.
    """
    # Normalize inputs
    image_dir = Path(image_dir).normalize()
    label_dir = Path(label_dir).normalize()
    output_json = Path(output_json).normalize()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Loop through each pair of image-label files
    image_files = [f for f in image_dir.glob("*") if f.is_image_file()]
    image_files = sorted(image_files)

    # Create remap dictionary
    if remap and isinstance(remap, (Path, str)):
        remap = load_yaml(path=remap)

    labels = []
    with create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence=enumerate(image_files),
            total=len(image_files),
            description=f"[bright_yellow]Processing",
        ):
            image_id = i
            imgsz = read_imgsz(image_file)

            # Load and convert bounding boxes to YOLO format
            label_file = label_dir / f"{image_file.stem}{K.LABEL_EXT}"
            bboxes: BBoxes = load_bbox(path=label_file, fmt=fmt, remap=remap, imgsz=imgsz)

            # Append labels
            if bboxes.is_empty:
                continue

            for b in bboxes:
                labels.append({
                    "image_id": image_id,
                    "category_id": b.class_id,
                    "bbox": [round(float(v), 32) for v in b.xywh(imgsz)],
                    "score": b.score,
                })

    # Write to JSON file
    with open(output_json.as_posix(), "w") as f:
        json.dump(labels, f, indent=None)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
