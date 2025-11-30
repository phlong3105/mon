#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for bounding boxes I/O operations.

This module provides functions to load and save bounding boxes in various
formats, including COCO, VOC, and YOLO.
"""

__all__ = [
    "load",
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
def _load_coco(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Loads COCO-format bounding boxes from a .json file.
    
    Args:
        path (Path): Label file path (one .json file for the dataset).
        remap (dict or box.Box, optional): Dictionary containing class remapping.
            Defaults to None.
        verbose (bool): Verbosity. Defaults to True.
        
    Returns:
        numpy.ndarray: Bounding boxes as a numpy array of shape (N, 7+).
    
    Raises:
        ValueError: If the file is not a valid .json file or contains no
            annotations.
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


def _load_voc(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Loads VOC-format bounding boxes from a .xml file.
    
    Note that this is a legacy format and is not recommended for new projects.

    Args:
        path (Path): Label file path (one .xml for each image).
        remap (dict or box.Box, optional): Dictionary containing class remapping.
            Defaults to None.
        verbose (bool): Verbosity. Defaults to True.
        
    Returns:
        numpy.ndarray: Bounding boxes as a numpy array of shape (N, 7+).
    """
    path = Path(path)
    if not path.is_xml_file(exist=True):
        if verbose:
            error_console.print(f"``path`` must be a valid .xmls file, got {path}.")
        return np.empty((0, 7), dtype=np.float32)

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
                float(obj.find("bndbox/ymax").text),
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
    return np.stack([x1, y1, x2, y2, 0, c] + rest, axis=-1)


def _load_yolo(
    path   : Path,
    remap  : dict | box.Box = None,
    verbose: bool = True
) -> np.ndarray:
    """Loads YOLO-format bounding boxes from a .txt file.
    
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
        path (Path): Label file path (one .txt for each image).
        remap (dict or box.Box, optional): Dictionary containing class remapping.
            Defaults to None.
        verbose (bool): Verbosity. Defaults to True.
        
    Returns:
        numpy.ndarray: Bounding boxes as a numpy array of shape (N, 7+).
    
    Raises:
        ValueError: If the file is not a valid .txt file or contains no
            bounding boxes.
    """
    path = Path(path)
    if not path.is_txt_file(exist=True):
        if verbose:
            error_console.print(f"``path`` must be a valid .txt file, got {path}.")
        return np.empty((0, 7), dtype=np.float32)

    with open(path, "r") as f:
        ls = f.readlines()
    ls = [l.strip().split(" ") for l in ls]
    ls = [l for l in ls if len(l) >= 4]

    if len(ls) == 0:
        if verbose:
            error_console.print(f"No bboxes found in {path}.")
        return np.empty((0, 7), dtype=np.float32)

    if len(ls[0]) == 4:
        # If no class ID, add a dummy class ID of 0
        ls = [[0] + l for l in ls]

    if remap and isinstance(remap, dict | box.Box):
        ls = [[remap[l[0]]] + l[1:] for l in ls]

    ls = np.array(ls, dtype=np.float32)
    c, cx_n, cy_n, w_n, h_n, a, *rest = ls.T
    if 0.0 < a < 1.0:  # No given angle, so ``a`` is a confidence score.
        rest = [a] + rest
        a    = 0.0
    return np.stack([cx_n, cy_n, w_n, h_n, a, c] + rest, axis=-1)


def load(
    path   : Path,
    fmt    : BBoxFormat,
    imgsz  : tuple[int, int],
    remap  : dict | box.Box = None,
    verbose: bool = False
) -> np.ndarray:
    """Loads bounding boxes from a file in the specified format.
    
    Args:
        path (Path): Label file path.
        fmt (BBoxFormat): Format of the bounding boxes to load.
        imgsz (tuple[int, int]): Image size as (width, height). Required
            when converting bounding boxes.
        remap (dict or box.Box, optional): Dictionary containing class remapping.
            Defaults to None.
        verbose (bool): Verbosity. Defaults to False.
        
    Returns:
        numpy.ndarray: Bounding boxes as a numpy array of shape (N, 7+).
        
    Raises:
        ValueError: If the specified format is not supported or if ``imgsz``
            is not provided when converting bounding boxes.
    """
    fmt = BBoxFormat(value=fmt)
    if fmt in BBoxFormat.conversion_codes():
        src_fmt = fmt.value.split("_to_")[0]
        src_fmt = BBoxFormat(value=src_fmt)
    else:
        src_fmt = fmt
        fmt     = None

    match src_fmt:
        case BBoxFormat.COCO | BBoxFormat.XYWH:
            bbox = _load_coco(path, remap, verbose)
        case BBoxFormat.VOC  | BBoxFormat.XYXY:
            bbox = _load_voc(path, remap, verbose)
        case BBoxFormat.YOLO | BBoxFormat.CXCYWHN:
            bbox = _load_yolo(path, remap, verbose)
        case _:
            raise ValueError(f"``src_fmt`` must be one of {BBoxFormat.formats()}, got {src_fmt}.")

    if fmt and imgsz is None:
        raise ValueError("``imgsz`` must be provided when converting bboxes.")

    if fmt:
        bbox = convert(bbox=bbox, fmt=fmt, imgsz=imgsz)

    return bbox
