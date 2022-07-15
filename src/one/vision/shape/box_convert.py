#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert box format.

References:
	https://github.com/pytorch/vision/blob/main/torchvision/ops/_box_convert.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from one.core import TensorOrArray
from one.core import upcast

__all__ = [
    "box_cxcyar_to_cxcyrh",
    "box_cxcyar_to_cxcywh",
    "box_cxcyar_to_cxcywhnorm",
    "box_cxcyar_to_xywh",
    "box_cxcyar_to_xyxy",
    "box_cxcyrh_to_cxcyar",
    "box_cxcyrh_to_cxcywh",
    "box_cxcyrh_to_cxcywh_norm",
    "box_cxcyrh_to_xywh",
    "box_cxcyrh_to_xyxy",
    "box_cxcywh_norm_to_cxcyar",
    "box_cxcywh_norm_to_cxcyrh",
    "box_cxcywh_norm_to_cxcywh",
    "box_cxcywh_norm_to_xywh",
    "box_cxcywh_norm_to_xyxy",
    "box_cxcywh_to_cxcyar",
    "box_cxcywh_to_cxcywh_norm",
    "box_cxcywh_to_xywh",
    "box_cxcywh_to_xyxy",
    "box_xywh_to_cxcyar",
    "box_xywh_to_cxcyrh",
    "box_xywh_to_cxcywh",
    "box_xywh_to_cxcywh_norm",
    "box_xywh_to_xyxy",
    "box_xyxy_to_cxcyar",
    "box_xyxy_to_cxcyrh",
    "box_xyxy_to_cxcywh",
    "box_xyxy_to_cxcywh_norm",
    "box_xyxy_to_xywh",
]


# MARK: - Functional

"""Coordination of bounding box's points.

(0, 0)              Image
      ---------------------------------- -> columns
      |                                |
      |        ----- -> x              |
      |        |   |                   |
      |        |   |                   |
      |        -----                   |
      |        |                       |
      |        V                       |
      |        y                       |
      ----------------------------------
      |                                 (n, m)
      V
     rows
"""


# MARK: cxcyar -> ...

def box_cxcyar_to_cxcyrh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, a, r) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w = torch.sqrt(a * r)
    h = a / w
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcyar_to_cxcywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, a, r) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w = torch.sqrt(a * r)
    h = a / w
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcyar_to_cxcywhnorm(box: TensorOrArray, height: int, width : int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, a, r) format to (cx, cy, w, h) norm
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        boxes (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w       = torch.sqrt(a * r)
    h       = (a / w)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    
    if isinstance(box, Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcyar_to_xywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, a, r) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format.
    """
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w = torch.sqrt(a * r)
    h = a / w
    x = cx - (w / 2.0)
    y = cy - (h / 2.0)
    
    if isinstance(box, Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
 
 
def box_cxcyar_to_xyxy(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, a, r) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w  = torch.sqrt(a * r)
    h  = a / w
    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)
    
    if isinstance(box, Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


# MARK: cxcyrh -> ...

def box_cxcyrh_to_cxcyar(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, r, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w = r * h
    a = w * h
    r = w / h
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_cxcyrh_to_cxcywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, r, h) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w = r * h

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcyrh_to_cxcywh_norm(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, r, h) format to (cx, cy, w, h) norm
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    box            = upcast(box)
    cx, cy, r, h, *_ = box.T
    w      = r * h
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height

    if isinstance(box, Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcyrh_to_xywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, r, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format.
    """
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w = r * h
    x = cx - w / 2.0
    y = cy - h / 2.0

    if isinstance(box, Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_cxcyrh_to_xyxy(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, r, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w = r * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    
    if isinstance(box, Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
   
   
# MARK: cxcywh ->

def box_cxcywh_to_cxcyar(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    a = w * h
    r = w / h
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_to_cxcyrh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    r = w / h
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_to_cxcywh_norm(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) norm format.
    """
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height

    if isinstance(box, Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_to_xywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format.
    """
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    x = cx - w / 2.0
    y = cy - h / 2.0
    
    if isinstance(box, Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_cxcywh_to_xyxy(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    
    if isinstance(box, Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

# MARK: cxcywh_norm ->

def box_cxcywh_norm_to_cxcyar(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) norm format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    box = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx = cx_norm * width
    cy = cy_norm * height
    a  = (w_norm * width) * (h_norm * height)
    r  = (w_norm * width) / (h_norm * height)

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_cxcywh_norm_to_cxcyrh(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) norm format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    box = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx = cx_norm * width
    cy = cy_norm * height
    r  = (w_norm * width) / (h_norm * height)
    h  = h_norm * height
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_norm_to_cxcywh(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) norm format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    box = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx = cx_norm * width
    cy = cy_norm * height
    w  = w_norm * width
    h  = h_norm * height
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_norm_to_xywh(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) norm format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format.
    """
    box = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    w = w_norm * width
    h = h_norm * height
    x = (cx_norm * width) - (w / 2.0)
    y = (cy_norm * height) - (h / 2.0)
    
    if isinstance(box, Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_cxcywh_norm_to_xyxy(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (cx, cy, w, h) norm format to (x1, y1,
    x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    box = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    x1 = width  * (cx_norm - w_norm / 2)
    y1 = height * (cy_norm - h_norm / 2)
    x2 = width  * (cx_norm + w_norm / 2)
    y2 = height * (cy_norm + h_norm / 2)
    
    if isinstance(box, Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


# MARK: xywh ->

def box_xywh_to_cxcyar(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    a  = w * h
    r  = w / h

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_xywh_to_cxcyrh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    r  = w / h

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_xywh_to_cxcywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_xywh_to_cxcywh_norm(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, w, h) norm
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx      = x + (w / 2.0)
    cy      = y + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height

    if isinstance(box, Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_xywh_to_xyxy(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    x2 = x + w
    y2 = y + h

    if isinstance(box, Tensor):
        return torch.stack((x, y, x2, y2), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x, y, x2, y2), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


# MARK: xyxy ->

def box_xyxy_to_cxcyar(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    a  = w * h
    r  = w / h

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
    

def box_xyxy_to_cxcyrh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    r  = w / h

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_xyxy_to_cxcywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)

    if isinstance(box, Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")


def box_xyxy_to_cxcywh_norm(box: TensorOrArray, height: int, width: int) -> TensorOrArray:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w,
    h) norm
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height

    if isinstance(box, Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
   

def box_xyxy_to_xywh(box: TensorOrArray) -> TensorOrArray:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (TensorOrArray[*, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (TensorOrArray[*, 4]):
            Boxes in (x, y, w, h) format.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w  = x2 - x1
    h  = y2 - y1
   
    if isinstance(box, Tensor):
        return torch.stack((x1, y1, w, h), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((x1, y1, w, h), -1)
    else:
        raise ValueError(f"box must be a `Tensor` or `np.ndarray`.")
