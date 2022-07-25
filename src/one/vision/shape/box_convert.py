#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert box format.

References:
	https://github.com/pytorch/vision/blob/main/torchvision/ops/_box_convert.py
"""

from __future__ import annotations

import inspect
import sys

import torch
from torch import Tensor

from one.core import assert_tensor_of_ndim
from one.core import upcast


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


def box_cxcyar_to_cxcyrh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, a, r) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    return torch.stack((cx, cy, r, h), -1)
   

def box_cxcyar_to_cxcywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, a, r) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    return torch.stack((cx, cy, w, h), -1)
    

def box_cxcyar_to_cxcywhnorm(box: Tensor, height: int, width : int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        boxes (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = (a / w)
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w / width
    h_norm           = h / height
    return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
   

def box_cxcyar_to_xywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, a, r) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    x                = cx - (w / 2.0)
    y                = cy - (h / 2.0)
    return torch.stack((x, y, w, h), -1)
    
    
def box_cxcyar_to_xyxy(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, a, r) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    x1               = cx - (w / 2.0)
    y1               = cy - (h / 2.0)
    x2               = cx + (w / 2.0)
    y2               = cy + (h / 2.0)
    return torch.stack((x1, y1, x2, y2), -1)
    

def box_cxcyrh_to_cxcyar(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, r, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    a                = w * h
    r                = w / h
    return torch.stack((cx, cy, a, r), -1)


def box_cxcyrh_to_cxcywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, r, h) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    return torch.stack((cx, cy, w, h), -1)


def box_cxcyrh_to_cxcywh_norm(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w  / width
    h_norm           = h  / height
    return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)


def box_cxcyrh_to_xywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, r, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    x                = cx - w / 2.0
    y                = cy - h / 2.0
    return torch.stack((x, y, w, h), -1)
    

def box_cxcyrh_to_xyxy(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, r, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    x1               = cx - w / 2.0
    y1               = cy - h / 2.0
    x2               = cx + w / 2.0
    y2               = cy + h / 2.0
    return torch.stack((x1, y1, x2, y2), -1)
   
   
def box_cxcywh_to_cxcyar(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, w, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    a                = w * h
    r                = w / h
    return torch.stack((cx, cy, a, r), -1)
    

def box_cxcywh_to_cxcyrh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    r                = w / h
    return torch.stack((cx, cy, r, h), -1)
   

def box_cxcywh_to_cxcywh_norm(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) norm format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w  / width
    h_norm           = h  / height
    return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
   

def box_cxcywh_to_xywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, w, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    x                = cx - w / 2.0
    y                = cy - h / 2.0
    return torch.stack((x, y, w, h), -1)
    

def box_cxcywh_to_xyxy(box: Tensor) -> Tensor:
    """Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format which will be converted.
        
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    assert_tensor_of_ndim(box, 2)
    box              = upcast(box)
    cx, cy, w, h, *_ = box.T
    x1               = cx - w / 2.0
    y1               = cy - h / 2.0
    x2               = cx + w / 2.0
    y2               = cy + h / 2.0
    return torch.stack((x1, y1, x2, y2), -1)
    

def box_cxcywh_norm_to_cxcyar(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                                  = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    a                                    = (w_norm * width) * (h_norm * height)
    r                                    = (w_norm * width) / (h_norm * height)
    return torch.stack((cx, cy, a, r), -1)
    

def box_cxcywh_norm_to_cxcyrh(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                                  = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    r                                    = (w_norm * width) / (h_norm * height)
    h                                    = h_norm * height
    return torch.stack((cx, cy, r, h), -1)
    

def box_cxcywh_norm_to_cxcywh(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                                  = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    w                                    = w_norm  * width
    h                                    = h_norm  * height
    return torch.stack((cx, cy, w, h), -1)
   

def box_cxcywh_norm_to_xywh(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                                  = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    w                                    = w_norm * width
    h                                    = h_norm * height
    x                                    = (cx_norm * width) - (w / 2.0)
    y                                    = (cy_norm * height) - (h / 2.0)
    return torch.stack((x, y, w, h), -1)
   
   
def box_cxcywh_norm_to_xyxy(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                                  = upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    x1                                   = width  * (cx_norm - w_norm / 2)
    y1                                   = height * (cy_norm - h_norm / 2)
    x2                                   = width  * (cx_norm + w_norm / 2)
    y2                                   = height * (cy_norm + h_norm / 2)
    return torch.stack((x1, y1, x2, y2), -1)


def box_xywh_to_cxcyar(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    assert_tensor_of_ndim(box, 2)
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    a              = w * h
    r              = w / h
    return torch.stack((cx, cy, a, r), -1)
    

def box_xywh_to_cxcyrh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    r              = w / h
    return torch.stack((cx, cy, r, h), -1)


def box_xywh_to_cxcywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x, y, w, h) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    return torch.stack((cx, cy, w, h), -1)
    

def box_xywh_to_cxcywh_norm(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    assert_tensor_of_ndim(box, 2)
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    cx_norm        = cx / width
    cy_norm        = cy / height
    w_norm         = w  / width
    h_norm         = h  / height
    return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)


def box_xywh_to_xyxy(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format.
    """
    assert_tensor_of_ndim(box, 2)
    box            = upcast(box)
    x, y, w, h, *_ = box.T
    x2             = x + w
    y2             = y + h
    return torch.stack((x, y, x2, y2), -1)
   

def box_xyxy_to_cxcyar(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, a, r) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    a                  = w * h
    r                  = w / h
    return torch.stack((cx, cy, a, r), -1)
   

def box_xyxy_to_cxcyrh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, r, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    r                  = w / h
    return torch.stack((cx, cy, r, h), -1)
    

def box_xyxy_to_cxcywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    return torch.stack((cx, cy, w, h), -1)
   

def box_xyxy_to_cxcywh_norm(box: Tensor, height: int, width: int) -> Tensor:
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
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
        height (int):
            Height of the image.
        width (int):
            Width of the image.
            
    Returns:
        box (Tensor[N, 4]):
            Boxes in (cx, cy, w, h) norm format.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    cx_norm            = cx / width
    cy_norm            = cy / height
    w_norm             = w  / width
    h_norm             = h  / height
    return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
   

def box_xyxy_to_xywh(box: Tensor) -> Tensor:
    """Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
           bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box (Tensor[N, 4]):
            Boxes in (x1, y1, x2, y2) format which will be converted.
       
    Returns:
        box (Tensor[N, 4]):
            Boxes in (x, y, w, h) format.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    return torch.stack((x1, y1, w, h), -1)
    

# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
