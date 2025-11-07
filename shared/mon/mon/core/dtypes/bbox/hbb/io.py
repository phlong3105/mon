#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HBB I/O operations.

Common Tasks:
    - Load bboxes from disk.
    - Save bboxes to disk.
    - Batch I/O.
    - Metadata handling.
"""

__all__ = [
    "load_hbb",
]

import json
import xml.etree.ElementTree as ET

import numpy as np
from box import box

from mon.core.console import error_console
from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from .processing import convert


# ----- Reading -----
def load_hbb_coco(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load COCO-format HBBs from a ``.json`` file.

    Args:
        path: Label file path (one ``.json`` file for each image).
        remap: Dictionary containing class remapping. Default: ``None``.
        verbose: Verbosity. Defaults is ``True``.
    """
    path = Path(path)
    if not path.is_json_file(exist=True):
        if verbose:
            error_console.print(f"``path`` must be a valid .json file, got {path}.")

    json_data = {}
    with open(path, "r") as f:
        json_data = json.load(f)

    info        = json_data.get("info",        {})
    licenses    = json_data.get("licenses",    [])
    categories  = json_data.get("categories",  [])
    images      = json_data.get("images",      [])
    annotations = json_data.get("annotations", [])

    if len(annotations) == 0:
        if verbose:
            error_console.print(f"No annotations found in {path}.")


def load_hbb_voc(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load VOC-format HBBs from a ``.xml`` file.

    Args:
        path: Label file path (one ``.xml`` file for each image).
        remap: Dictionary containing class remapping. Default: ``None``.
        verbose: Verbosity. Defaults is ``True``.
    """
    path = Path(path)
    if not path.is_xml_file(exist=True):
        if verbose:
            error_console.print(f"``path`` must be a valid .xmls file, got {path}.")
        return np.empty((0, 6), dtype=np.float32)

    tree = ET.parse(str(path))
    root = tree.getroot()

    xml_data = {
        "filename" : "",
        "width"    : 0,
        "height"   : 0,
        "depth"    : 0,
        "objects"  : [],
        "segmented": 0
    }

    # Extract image metadata
    xml_data["filename"] = root.find("filename").text
    size = root.find("size")
    xml_data["width"]    = int(size.find("width").text)
    xml_data["height"]   = int(size.find("height").text)
    xml_data["depth"]    = int(size.find("depth").text)

    # Extract segmented flag (0 or 1)
    segmented = root.find("segmented")
    if segmented is not None:
        xml_data["segmented"] = int(segmented.text)

    # Extract objects
    for obj in root.findall("object"):
        obj_data = {
            "name": obj.find("name").text,
            "bbox": [
                float(obj.find("bndbox/xmin").text),
                float(obj.find("bndbox/ymin").text),
                float(obj.find("bndbox/xmax").text),
                float(obj.find("bndbox/ymax").text)
            ],
            "difficult": int(obj.find("difficult").text) if obj.find("difficult") is not None else 0,
            "truncated": int(obj.find("truncated").text) if obj.find("truncated") is not None else 0,
            "pose"     :     obj.find("pose").text       if obj.find("pose")      is not None else None
        }
        xml_data["objects"].append(obj_data)

    # Extract bounding boxes
    bs = []
    for obj in xml_data["objects"]:
        bs.append([obj["name"]] + obj["bbox"])

    if remap and isinstance(remap, dict | box.Box):
        bs = [[remap[int(b[0])]] + b[1:] for b in bs]

    bs = np.array(bs, dtype=np.float32)
    c, x1, y1, x2, y2, *rest = bs.T
    return np.stack([x1, y1, x2, y2, c] + rest, axis=-1)


def load_hbb_yolo(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Load YOLO-format HBBs from a ``.txt`` file.

    Each line in the file should contain:
        <class_id> <center_x> <center_y> <width> <height> | Optional: <confidence>
    where:
        - ``class_id`` is the class index (0-based).
        - ``x_center``, ``y_center``, ``width``, and ``height`` are normalized values
            relative to the image dimensions.
        - ``confidence`` is an optional value representing the confidence score.

    Args:
        path: Label file path (one ``.txt`` for each image).
        remap: Dictionary containing class remapping. Default: ``None``.
        verbose: Verbosity. Defaults is ``True``.
    """
    path = Path(path)
    if not path.is_txt_file(exist=True):
        if verbose:
            error_console.print(f"``path`` must be a valid .txt file, got {path}.")
        return np.empty((0, 6), dtype=np.float32)

    with open(path, "r") as f:
        ls = f.readlines()
    ls = [l.strip().split(" ") for l in ls]
    ls = [l for l in ls if len(l) >= 4]

    if len(ls) == 0:
        if verbose:
            error_console.print(f"No HBBs found in {path}.")
        return np.empty((0, 6), dtype=np.float32)

    if len(ls[0]) == 4:
        # If no class ID, add a dummy class ID of 0
        ls = [[0] + l for l in ls]

    if remap and isinstance(remap, dict | box.Box):
        ls = [[remap[l[0]]] + l[1:] for l in ls]

    ls = np.array(ls, dtype=np.float32)
    c, cx_n, cy_n, w_n, h_n, *rest = ls.T
    return np.stack([cx_n, cy_n, w_n, h_n, c] + rest, axis=-1)


def load_hbb(
    path   : Path,
    fmt    : BBoxFormat,
    imgsz  : tuple[int, int],
    remap  : dict | box.Box = None,
    verbose: bool = False
) -> np.ndarray:
    """Load HBBs from a label file.

    Args:
        path: Label file path.
        fmt: Bounding box format of the label file.
        imgsz: Image size as a ``tuple`` of :math:`(H, W)`.
        remap: Dictionary containing class remapping. Default: ``None``.
        verbose: Verbosity. Defaults is ``False``.

    Returns:
        HBBs as a ``numpy.ndarray`` in [N, 4+], output format varies by code.

    Raises:
        ValueError: If ``format`` is invalid.
    """
    fmt = BBoxFormat.from_value(value=fmt)
    if fmt in BBoxFormat.conversion_codes():
        src_fmt = fmt.value.split("_to_")[0]
        src_fmt = BBoxFormat.from_value(value=src_fmt)
    else:
        src_fmt = fmt
        fmt     = None

    match src_fmt:
        case BBoxFormat.COCO | BBoxFormat.XYWH:
            bbox = load_hbb_coco(path, remap, verbose)
        case BBoxFormat.VOC  | BBoxFormat.XYXY:
            bbox = load_hbb_voc(path, remap, verbose)
        case BBoxFormat.YOLO | BBoxFormat.CXCYWHN:
            bbox = load_hbb_yolo(path, remap, verbose)
        case _:
            raise ValueError(f"``src_fmt`` must be one of {BBoxFormat.formats()}, got {src_fmt}.")

    if fmt and imgsz is None:
        raise ValueError("[imgsz] must be provided when converting HBBs.")

    if fmt:
        bbox = convert(bbox=bbox, fmt=fmt, imgsz=imgsz)

    return bbox


# ----- Writing -----
