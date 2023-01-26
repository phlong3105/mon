#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.coreimage.geometry.box` package implements geometry functions for
bounding box.
"""

from __future__ import annotations

__all__ = [
    "affine_box", "box_cxcyar_to_cxcyrh", "box_cxcyar_to_cxcywh",
    "box_cxcyar_to_cxcywhnorm", "box_cxcyar_to_xywh", "box_cxcyar_to_xyxy",
    "box_cxcyrh_to_cxcyar", "box_cxcyrh_to_cxcywh", "box_cxcyrh_to_cxcywh_norm",
    "box_cxcyrh_to_xywh", "box_cxcyrh_to_xyxy", "box_cxcywh_norm_to_cxcyar",
    "box_cxcywh_norm_to_cxcyrh", "box_cxcywh_norm_to_cxcywh",
    "box_cxcywh_norm_to_xywh", "box_cxcywh_norm_to_xyxy",
    "box_cxcywh_to_cxcyar", "box_cxcywh_to_cxcyrh", "box_cxcywh_to_cxcywh_norm",
    "box_cxcywh_to_xywh", "box_cxcywh_to_xyxy", "box_xywh_to_cxcyar",
    "box_xywh_to_cxcyrh", "box_xywh_to_cxcywh", "box_xywh_to_cxcywh_norm",
    "box_xywh_to_xyxy", "box_xyxy_to_cxcyar", "box_xyxy_to_cxcyrh",
    "box_xyxy_to_cxcywh", "box_xyxy_to_cxcywh_norm", "box_xyxy_to_xywh",
    "clip_box", "compute_box_area", "compute_box_intersection_union",
    "compute_box_iou", "compute_box_iou_old", "generate_box", "get_box_center",
    "get_box_corners", "get_box_corners_points", "get_enclosing_box",
    "horizontal_flip_box", "horizontal_translate_box", "nms", "rotate_box",
    "scale_box", "scale_box_original", "shear_box", "translate_box",
    "vertical_flip_box", "vertical_translate_box",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.coreimage import util
from mon.coreimage.typing import Floats, Ints


# region Affine Transform

def affine_box(
    box       : torch.Tensor,
    image_size: Ints,
    angle     : float | int,
    translate : Ints,
    scale     : float | int,
    shear     : Floats,
    center    : Ints  | None = None,
    drop_ratio: float        = 0.0,
) -> torch.Tensor:
    """Apply an affine transformation on the bounding box.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
    
    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The image size of shape [H, W].
        angle: A rotation angle in degrees, between -180 and 180, clockwise
            direction.
        translate: horizontal and vertical translations (post-rotation
            translation).
        scale: An overall scale.
        shear: A shear angle has a value in degrees, between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed box of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    
    assert isinstance(angle, float | int)
    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, int | float):
        translate = [translate, translate]
    if isinstance(translate, tuple):
        translate = list(translate)
    assert isinstance(translate, list | tuple) and len(translate) == 2
    
    if isinstance(scale, int):
        scale = float(scale)
    assert isinstance(scale, int | float) and scale >= 0.0

    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    assert isinstance(shear, list | tuple) and len(shear) == 2
    
    h, w   = util.to_size(image_size)
    center = (h * 0.5, w * 0.5) if center is None else center
    center = tuple(center[::-1])
    angle  = -angle
    r      = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    t      = translate
    s      = [core.math.radians(-shear[0]), core.math.radians(-shear[1])]
    m      = np.float32([[r[0, 0]       , s[0] + r[0, 1], r[0, 2] + t[0] + (-s[0] * center[1])],
                         [s[1] + r[1, 0], r[1, 1]       , r[1, 2] + t[1] + (-s[1] * center[0])],
                         [0             , 0             , 1]])
    m      = torch.from_numpy(m).to(torch.double).to(box.device)
    
    box       = box.clone()
    n         = len(box)
    xy        = torch.ones((n * 4, 3), dtype=box.dtype).to(box.device)
    xy[:, :2] = box[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy        = xy @ m.T  # Transform
    xy        = xy[:, :2].reshape(n, 8)
    
    x         = xy[:, [0, 2, 4, 6]]
    y         = xy[:, [1, 3, 5, 7]]
    x1        = torch.min(x, 1, keepdim=True).values
    y1        = torch.min(y, 1, keepdim=True).values
    x2        = torch.max(x, 1, keepdim=True).values
    y2        = torch.max(y, 1, keepdim=True).values
    xy        = torch.cat((x1, y1, x2, y2)).reshape(4, n).T
    box       = clip_box(
        box        = xy,
        image_size = image_size,
        drop_ratio = drop_ratio,
    )
    return box


def clip_box(
    box       : torch.Tensor,
    image_size: Ints,
    drop_ratio: float = 0.0,
) -> torch.Tensor:
    """Clip bounding boxes to an image size and removes the bounding boxes,
    which lose too much area as a result of the augmentation.
    
    Args:
        box: Bounding boxes of shape [N, 4], They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: An image size of shape [H, W].
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Clipped bounding boxes of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    h, w = util.to_size(image_size)
    area = compute_box_area(box)

    box = box.clone()
    box[:, 0].clamp_(0, w)  # x1
    box[:, 1].clamp_(0, h)  # y1
    box[:, 2].clamp_(0, w)  # x2
    box[:, 3].clamp_(0, h)  # y2
    delta_area = ((area - compute_box_area(box)) / area)
    mask       = (delta_area < (1 - drop_ratio)).to(torch.int)
    box        = box[mask == 1, :]
    return box


def horizontal_flip_box(
    box         : torch.Tensor,
    image_center: torch.Tensor
) -> torch.Tensor:
    """Horizontally flip boxes, which are specified by their normalized
    (cx, cy, w, h) coordinates.
    
    Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box: Bounding boxes of shape [N, 4] to be flipped.
        image_center: The center of the image.
        
    Returns:
        Flipped bounding boxes of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    assert isinstance(image_center, torch.Tensor) and image_center.ndim == 1
    box             = box.clone()
    box[:, [0, 2]] += 2 * (image_center[[0, 2]] - box[:, [0, 2]])
    box_w           = abs(box[:, 0] - box[:, 2])
    box[:, 0]      -= box_w
    box[:, 2]      += box_w
    return box


def horizontal_translate_box(
    box       : torch.Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes horizontally.

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The original image size.
        magnitude: A horizontal translation magnitude.
        center: The center of affine transformation. If None, use the center of the
            image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    box = translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [magnitude, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return box


def rotate_box(
    box       : torch.Tensor,
    image_size: Ints,
    angle     : float,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Rotate the bounding box by the given magnitude.

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The original image size.
        angle: An angle to rotate the bounding box.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
            
    Returns:
        A translated boxes of shape [N, 4].
    """
    box = affine_box(
        box        = box,
        image_size = image_size,
        angle      = angle,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return box


def scale_box(
    box       : torch.Tensor,
    cur_size  : Ints,
    new_size  : Ints   | None = None,
    factor    : Floats | None = (1.0, 1.0),
    keep_shape: bool          = False,
    drop_ratio: float         = 0.0,
) -> torch.Tensor:
    """Scale bounding boxes coordinates by the given factor or by inferring
    from current image size and new size.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size: The current image size.
        new_size: A new image size. Defaults to None.
        factor: A desired scaling factor in each direction. If scalar, the value
            is used for both the vertical and horizontal direction. Defaults to
            (1.0, 1.0).
        keep_shape: When True, translate the scaled bounding boxes. Defaults to
            False.
        drop_ratio: If a fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
            
    Returns:
        A scaled bounding boxes of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    h0, w0 = util.to_size(cur_size)  # H, W
    
    if new_size is not None:
        h1, w1                 = util.to_size(new_size)  # H, W
        factor_ver, factor_hor = float(h1 / h0), float(w1 / w0)
    elif isinstance(factor, float):
        factor_ver = factor_hor = factor
        h1, w1     = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    else:
        factor_ver, factor_hor = factor
        h1, w1                 = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    
    box         = box.clone()
    box[:, :4] *= [factor_hor, factor_ver, factor_hor, factor_ver]
    box         = clip_box(box=box, image_size=(h1, w1), drop_ratio=drop_ratio)
    if keep_shape and (h0 * w0) >= (h1 * w1):
        hor = int(abs(w0 - w1) / 2)
        ver = int(abs(h0 - h1) / 2)
        box = translate_box(
            box        = box,
            magnitude  = (hor, ver),
            image_size = (h1, w1),
        )
    return box


def scale_box_original(
    box       : torch.Tensor,
    cur_size  : Ints,
    new_size  : Ints,
    ratio_pad = None,
) -> torch.Tensor:
    """Scale bounding boxes coordinates (from the :param:`cur_size`) to the
    :param:`new_size`.

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size: The current size.
        new_size: A new size.
        ratio_pad: Defaults to None.
        
    Returns:
        Scaled bounding boxes of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    cur_size = util.to_size(cur_size)
    new_size = util.to_size(new_size)
    
    if ratio_pad is None:  # Calculate from new_size
        gain = min(cur_size[0] / new_size[0],
                   cur_size[1] / new_size[1])  # gain  = old / new
        pad  = (cur_size[1] - new_size[1] * gain) / 2, \
               (cur_size[0] - new_size[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad  = ratio_pad[1]

    box             = box.clone()
    box[:, [0, 2]] -= pad[0]  # x padding
    box[:, [1, 3]] -= pad[1]  # y padding
    box[:, :4]     /= gain
    box             = clip_box(box=box, image_size=new_size)
    return box


def shear_box(
    box       : torch.Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Shear bounding boxes.
    
    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The original image size.
        magnitude: A shear magnitude.
        center: The center of affine transformation. If None, use the center of the
            image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Sheared bounding boxes of shape [N, 4].
    """
    box = affine_box(
        box        = box,
        image_size = image_size,
        angle      = 0.0,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = magnitude,
        center     = center,
        drop_ratio = drop_ratio,
    )
    return box


def translate_box(
    box       : torch.Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes.

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The original image size.
        magnitude: A translation magnitude.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    box = affine_box(
        box        = box,
        image_size = image_size,
        angle      = 0.0,
        translate  = magnitude,
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return box
    

def vertical_flip_box(
    box         : torch.Tensor,
    image_center: torch.Tensor,
) -> torch.Tensor:
    """Flip bounding boxes vertically, which are specified by their (cx, cy, w,
    h) norm coordinates.
	
	Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box: Bounding boxes of shape [N, 4] to be flipped.
        image_center: The center of the image.
        
    Returns:
        Flipped boxes of shape [N, 4].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box             = box.clone()
    box[:, [1, 3]] += 2 * (image_center[[0, 2]] - box[:, [1, 3]])
    box_h           = abs(box[:, 1] - box[:, 3])
    box[:, 1]      -= box_h
    box[:, 3]      += box_h
    return box


def vertical_translate_box(
    box       : torch.Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes in vertical direction.

    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
        x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size: The original image size.
        magnitude: A vertically translation.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding box left in the image after
            being clipped is less than :param:`drop_ratio` the bounding box is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    box = translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [0, magnitude],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return box

# endregion


# region Box Format Conversion

def box_cxcyar_to_cxcyrh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, a, r) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, a, r) format.
        
    Returns:
        Bounding boxes in (cx, cy, r, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    box              = torch.stack((cx, cy, r, h), -1)
    return box


def box_cxcyar_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, a, r) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, a, r) format.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    box              = torch.stack((cx, cy, w, h), -1)
    return box


def box_cxcyar_to_cxcywhnorm(
    box   : torch.Tensor,
    height: int,
    width : int
) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, a, r) format to (cx, cy, w, h)
    norm format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, a, r) format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) norm format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = (a / w)
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w / width
    h_norm           = h / height
    box              = torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    return box


def box_cxcyar_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, a, r) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, a, r) format.
        
    Returns:
        Bounding boxes in (x, y, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    x                = cx - (w / 2.0)
    y                = cy - (h / 2.0)
    box              = torch.stack((x, y, w, h), -1)
    return box

    
def box_cxcyar_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, a, r) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, a, r) format.
        
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, a, r, *_ = box.T
    w                = torch.sqrt(a * r)
    h                = a / w
    x1               = cx - (w / 2.0)
    y1               = cy - (h / 2.0)
    x2               = cx + (w / 2.0)
    y2               = cy + (h / 2.0)
    box              = torch.stack((x1, y1, x2, y2), -1)
    return box


def box_cxcyrh_to_cxcyar(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, r, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, r, h) format.
        
    Returns:
        Bounding boxes in (cx, cy, a, r) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    a                = w * h
    r                = w / h
    box              = torch.stack((cx, cy, a, r), -1)
    return box


def box_cxcyrh_to_cxcywh(box: torch.Tensor) ->torch. Tensor:
    """Convert bounding boxes from (cx, cy, r, h) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, r, h) format.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    box              = torch.stack((cx, cy, w, h), -1)
    return box


def box_cxcyrh_to_cxcywh_norm(
    box   : torch.Tensor,
    height: int,
    width : int
) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, r, h) format to (cx, cy, w, h)
    norm format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, r, h) format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        box: Bounding boxes in (cx, cy, w, h) norm format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w  / width
    h_norm           = h  / height
    box              = torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    return box


def box_cxcyrh_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, r, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, r, h) format.
        
    Returns:
        Bounding boxes in (x, y, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    x                = cx - w / 2.0
    y                = cy - h / 2.0
    box              = torch.stack((x, y, w, h), -1)
    return box


def box_cxcyrh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, r, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, r, h) format.
        
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, r, h, *_ = box.T
    w                = r * h
    x1               = cx - w / 2.0
    y1               = cy - h / 2.0
    x2               = cx + w / 2.0
    y2               = cy + h / 2.0
    box              = torch.stack((x1, y1, x2, y2), -1)
    return box


def box_cxcywh_to_cxcyar(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) format.
        
    Returns:
        Bounding boxes in (cx, cy, a, r) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, w, h, *_ = box.T
    a                = w * h
    r                = w / h
    box              = torch.stack((cx, cy, a, r), -1)
    return box


def box_cxcywh_to_cxcyrh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) format.
        
    Returns:
        Bounding boxes in (cx, cy, r, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, w, h, *_ = box.T
    r                = w / h
    box              = torch.stack((cx, cy, r, h), -1)
    return box


def box_cxcywh_to_cxcywh_norm(
    box   : torch.Tensor,
    height: int,
    width : int
) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, r, h) norm format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, w, h, *_ = box.T
    cx_norm          = cx / width
    cy_norm          = cy / height
    w_norm           = w  / width
    h_norm           = h  / height
    box              = torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    return box
    

def box_cxcywh_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) format.
        
    Returns:
        Bounding boxes in (x, y, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, w, h, *_ = box.T
    x                = cx - w / 2.0
    y                = cy - h / 2.0
    box              = torch.stack((x, y, w, h), -1)
    return box
    

def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) format.
        
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box              = util.upcast(box)
    cx, cy, w, h, *_ = box.T
    x1               = cx - w / 2.0
    y1               = cy - h / 2.0
    x2               = cx + w / 2.0
    y2               = cy + h / 2.0
    box              = torch.stack((x1, y1, x2, y2), -1)
    return box


def box_cxcywh_norm_to_cxcyar(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) norm format to (cx, cy, a,
    r) format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, a, r) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                                  = util.upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    a                                    = (w_norm * width) * (h_norm * height)
    r                                    = (w_norm * width) / (h_norm * height)
    box                                  = torch.stack((cx, cy, a, r), -1)
    return box


def box_cxcywh_norm_to_cxcyrh(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) norm format to (cx, cy, r,
    h) format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, r, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                                  = util.upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    r                                    = (w_norm * width) / (h_norm * height)
    h                                    = h_norm * height
    box                                  = torch.stack((cx, cy, r, h), -1)
    return box


def box_cxcywh_norm_to_cxcywh(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) norm format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                                  = util.upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    cx                                   = cx_norm * width
    cy                                   = cy_norm * height
    w                                    = w_norm  * width
    h                                    = h_norm  * height
    box                                  = torch.stack((cx, cy, w, h), -1)
    return box


def box_cxcywh_norm_to_xywh(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) norm format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (x, y, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                                  = util.upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    w                                    = w_norm * width
    h                                    = h_norm * height
    x                                    = (cx_norm * width) - (w / 2.0)
    y                                    = (cy_norm * height) - (h / 2.0)
    box                                  = torch.stack((x, y, w, h), -1)
    return box
   
   
def box_cxcywh_norm_to_xyxy(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (cx, cy, w, h) norm format to (x1, y1, x2,
    y2) format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (cx, cy, w, h) norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                                  = util.upcast(box)
    cx_norm, cy_norm, w_norm, h_norm, *_ = box.T
    x1                                   = width  * (cx_norm - w_norm / 2)
    y1                                   = height * (cy_norm - h_norm / 2)
    x2                                   = width  * (cx_norm + w_norm / 2)
    y2                                   = height * (cy_norm + h_norm / 2)
    box                                  = torch.stack((x1, y1, x2, y2), -1)
    return box


def box_xywh_to_cxcyar(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) format to (cx, cy, a, r) format.
    
    (cx, cy) refers to the center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x, y, w, h) format.
       
    Returns:
        Bounding boxes in (cx, cy, a, r) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box            = util.upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    a              = w * h
    r              = w / h
    box            =  torch.stack((cx, cy, a, r), -1)
    return box


def box_xywh_to_cxcyrh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) format to (cx, cy, r, h) format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x, y, w, h) format.
       
    Returns:
        Bounding boxes in (cx, cy, r, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box            = util.upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    r              = w / h
    box            = torch.stack((cx, cy, r, h), -1)
    return box
    

def box_xywh_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) format to (cx, cy, w, h) format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x, y, w, h) format.
       
    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box            = util.upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    box            =  torch.stack((cx, cy, w, h), -1)
    return box


def box_xywh_to_cxcywh_norm(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) format to (cx, cy, w, h) norm
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to the normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (x, y, w, h) format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) norm format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box            = util.upcast(box)
    x, y, w, h, *_ = box.T
    cx             = x + (w / 2.0)
    cy             = y + (h / 2.0)
    cx_norm        = cx / width
    cy_norm        = cy / height
    w_norm         = w  / width
    h_norm         = h  / height
    box            = torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    return box


def box_xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x, y, w, h) format.
       
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box            = util.upcast(box)
    x, y, w, h, *_ = box.T
    x2             = x + w
    y2             = y + h
    box            = torch.stack((x, y, x2, y2), -1)
    return box


def box_xyxy_to_cxcyar(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, a, r)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format.
       
    Returns:
        Bounding boxes in (cx, cy, a, r) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    a                  = w * h
    r                  = w / h
    box                = torch.stack((cx, cy, a, r), -1)
    return box
    

def box_xyxy_to_cxcyrh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, r, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format.
       
    Returns:
        Bounding boxes in (cx, cy, r, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    r                  = w / h
    box                = torch.stack((cx, cy, r, h), -1)
    return box


def box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format.
       
    Returns:
        Bounding boxes in (cx, cy, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    box                = torch.stack((cx, cy, w, h), -1)
    return box


def box_xyxy_to_cxcywh_norm(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h)
    norm format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
    _norm refers to the normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in (cx, cy, w, h) norm format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    cx                 = x1 + (w / 2.0)
    cy                 = y1 + (h / 2.0)
    cx_norm            = cx / width
    cy_norm            = cy / height
    w_norm             = w  / width
    h_norm             = h  / height
    box                = torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    return box


def box_xyxy_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h)
    format.
    
    (cx, cy) refers to center of bounding box.
    (a, r) refers to area (width * height) and aspect ratio (width / height) of
        bounding box.
    (w, h) refers to width and height of bounding box.
   
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format.
       
    Returns:
        Bounding boxes in (x, y, w, h) format.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    w                  = x2 - x1
    h                  = y2 - y1
    box                = torch.stack((x1, y1, w, h), -1)
    return box

# endregion


# region Box Property

def compute_box_area(box: torch.Tensor) -> torch.Tensor:
    """Compute the area of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box: Bounding boxes in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
    
    Returns:
        The area for each box.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box                = util.upcast(box)
    x1, y1, x2, y2, *_ = box.T
    area               = (x2 - x1) * (y2 - y1)
    return area


def compute_box_intersection_union(
    box1: torch.Tensor, box2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the intersection and union of two sets of boxes. Both sets of
    boxes are expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
    `0 <= y1 < y2`.
    
    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.
    
    Args:
        box1: The first set of boxes of shape [N, 4].
        box2: The second set of boxes of shape [N, 4].
        
    Returns:
        Intersection.
        Union.
    """
    assert isinstance(box1, torch.Tensor) and box1.ndim == 2
    assert isinstance(box2, torch.Tensor) and box2.ndim == 2
    area1 = compute_box_area(box1)
    area2 = compute_box_area(box2)
    lt    = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb    = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    wh    = util.upcast(rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter, union


def compute_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Return the intersection-over-union (Jaccard index) between two sets of
    boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Args:
        box1: The first set of boxes of shape [N, 4].
        box2: The second set of boxes of shape [M, 4].
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element
        in boxes1 and boxes2.
    """
    inter, union = compute_box_intersection_union(box1, box2)
    iou          = inter / union
    return iou


def compute_box_iou_old(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """From SORT: Compute IOU between two sets of boxes.
    
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.

    Args:
        box1: The first set of boxes of shape [N, 4].
        box2: The second set of boxes of shape [M, 4].
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element in
        boxes1 and boxes2.
    """
    assert isinstance(box1, torch.Tensor) and box1.ndim == 2
    assert isinstance(box2, torch.Tensor) and box2.ndim == 2
    box1 = torch.unsqueeze(box1, 1)
    box2 = torch.unsqueeze(box2, 0)
    xx1  = torch.maximum(box1[..., 0], box2[..., 0])
    yy1  = torch.maximum(box1[..., 1], box2[..., 1])
    xx2  = torch.minimum(box1[..., 2], box2[..., 2])
    yy2  = torch.minimum(box1[..., 3], box2[..., 3])
    w    = torch.maximum(torch.Tensor(0.0), xx2 - xx1)
    h    = torch.maximum(torch.Tensor(0.0), yy2 - yy1)
    wh   = w * h
    iou  = wh / ((box1[..., 2] - box1[..., 0]) *
                 (box1[..., 3] - box1[..., 1]) +
                 (box2[..., 2] - box2[..., 0]) *
                 (box2[..., 3] - box2[..., 1]) - wh)
    return iou


def generate_box(
    x_start: torch.Tensor,
    y_start: torch.Tensor,
    width  : torch.Tensor,
    height : torch.Tensor
) -> torch.Tensor:
    """Generate 2D bounding boxes according to the provided start coords, width
    and height.

    Args:
        x_start: Tensor containing the x coordinates of the bounding boxes to be
            extracted. Shape must be a scalar image or [B].
        y_start: Tensor containing the y coordinates of the bounding boxes to be
            extracted. Shape must be a scalar image or [B].
        width: Widths of the masked image. Shape must be a scalar image or [B].
        height: Heights of the masked image. Shape must be a scalar image or
            [B].

    Returns:
        Bounding box.

    Examples:
        >>> x_start = torch.Tensor([0, 1])
        >>> y_start = torch.Tensor([1, 0])
        >>> width   = torch.Tensor([5, 3])
        >>> height  = torch.Tensor([7, 4])
        >>> generate_box(x_start, y_start, width, height)
        image([[[0, 1],
                 [4, 1],
                 [4, 7],
                 [0, 7]],
        <BLANKLINE>
                [[1, 0],
                 [3, 0],
                 [3, 3],
                 [1, 3]]])
    """
    if not (x_start.shape == y_start.shape and x_start.dim() in [0, 1]):
        raise AssertionError(f"`x_start` and `y_start` must be a scalar or "
                             f"[B,]. But got: {x_start}, {y_start}.")
    if not (width.shape == height.shape and width.dim() in [0, 1]):
        raise AssertionError(f"`width` and `height` must be a scalar or "
                             f"[B,]. But got: {width}, {height}.")
    if not x_start.dtype == y_start.dtype == width.dtype == height.dtype:
        raise AssertionError(
            f"All tensors must be in the same dtype. But got: "
            f"`x_start`({x_start.dtype}), `y_start`({x_start.dtype}), "
            f"`width`({width.dtype}), `height`({height.dtype})."
        )
    if not x_start.device == y_start.device == width.device == height.device:
        raise AssertionError(
            f"All tensors must be in the same device. But got: "
            f"`x_start`({x_start.device}), `y_start`({x_start.device}), "
            f"`width`({width.device}), `height`({height.device})."
        )

    box = (
        torch.tensor(
            [[[0, 0], [0, 0], [0, 0], [0, 0]]],
            device=x_start.device, dtype=x_start.dtype
        ).repeat(1 if x_start.dim() == 0 else len(x_start), 1, 1)
    )
    box[:, :, 0] += x_start.view(-1, 1)
    box[:, :, 1] += y_start.view(-1, 1)
    box[:, 1, 0] += width - 1
    box[:, 2, 0] += width - 1
    box[:, 2, 1] += height - 1
    box[:, 3, 1] += height - 1

    return box


def get_box_center(box: torch.Tensor) -> torch.Tensor:
    """Compute the center of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        The center for each box of shape [N, 2].
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    box          = util.upcast(box)
    box          = box_xyxy_to_cxcywh(box)
    cx, cy, w, h = box.T
    center       = torch.stack((cx, cy), -1)
    return center
    

def get_box_corners(box: torch.Tensor) -> torch.Tensor:
    """Get corners of bounding boxes.
    
    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        A tensor of shape `N x 8` containing N bounding boxes each described by
        their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    width   = (box[:, 2] - box[:, 0]).reshape(-1, 1)
    height  = (box[:, 3] - box[:, 1]).reshape(-1, 1)
    x1      = box[:, 0].reshape(-1, 1)
    y1      = box[:, 1].reshape(-1, 1)
    x2      = x1 + width
    y2      = y1
    x3      = x1
    y3      = y1 + height
    x4      = box[:, 2].reshape(-1, 1)
    y4      = box[:, 3].reshape(-1, 1)
    corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners


def get_box_corners_points(box: torch.Tensor) -> torch.Tensor:
    """Get corners of bounding boxes as points.
    
    Args:
        box: Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        Corners.
    """
    assert isinstance(box, torch.Tensor)
    if box.ndim == 2:
        width  = (box[:, 2] - box[:, 0]).reshape(-1, 1)
        height = (box[:, 3] - box[:, 1]).reshape(-1, 1)
        x1     = box[:, 0].reshape(-1, 1)
        y1     = box[:, 1].reshape(-1, 1)
        x2     = x1 + width
        y2     = y1
        x3     = x1
        y3     = y1 + height
        x4     = box[:, 2].reshape(-1, 1)
        y4     = box[:, 3].reshape(-1, 1)
    else:
        width  = box[2] - box[0]
        height = box[3] - box[1]
        x1     = box[0]
        y1     = box[1]
        x2     = x1 + width
        y2     = y1
        x3     = x1
        y3     = y1 + height
        x4     = box[2]
        y4     = box[3]
    corners = torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return corners
    

def get_enclosing_box(box: torch.Tensor) -> torch.Tensor:
    """Get an enclosing box for rotated corners of a bounding box.
    
    Args:
        box: Bounding of shape [N, 8], containing N bounding boxes each
            described by their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Returns:
        Bounding boxes of shape [N, 4]. They are expected to be in (x1, y1, x2,
            y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """
    assert isinstance(box, torch.Tensor) and box.ndim == 2
    x_    = box[:, [0, 2, 4, 6]]
    y_    = box[:, [1, 3, 5, 7]]
    x1    = torch.min(x_, 1).reshape(-1, 1)
    y1    = torch.min(y_, 1).reshape(-1, 1)
    x2    = torch.max(x_, 1).reshape(-1, 1)
    y2    = torch.max(y_, 1).reshape(-1, 1)
    final = torch.stack((x1, y1, x2, y2, box[:, 8:]))
    return final


def nms(
    box          : torch.Tensor,
    scores       : torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """Performs non-maxima suppression (NMS) on a given image of bounding boxes
    according to the intersection-over-union (IoU).
    
    NMS iteratively removes lower scoring boxes which have an IoU greater than
    `iou_threshold` with another (higher scoring) box.
    
    If multiple boxes have the exact same score and satisfy the IoU criterion
    with respect to a reference box, the selected box is not guaranteed to be
    the same between CPU and GPU. This is similar to the behavior of argsort in
    PyTorch when repeated values are present.
    
    Args:
        box: Bounding boxes of shape [N, 4] to perform NMS on. They are expected
            to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        scores: Scores for each one of the boxes.
        iou_threshold: Discards all overlapping boxes with
            iou > :param:`iou_threshold`.

    Return:
        Indices of the elements that have been kept by NMS, sorted in decreasing
        order of scores

    Example:
        >>> boxes  = torch.Tensor([
        ...     [10., 10., 20., 20.],
        ...     [15., 5., 15., 25.],
        ...     [100., 100., 200., 200.],
        ...     [100., 100., 200., 200.]])
        >>> scores = torch.Tensor([0.9, 0.8, 0.7, 0.9])
        >>> nms(box, scores, iou_threshold=0.8)
        image([0, 3, 1])
    """
    assert isinstance(box, torch.Tensor)    and box.ndim == 2
    assert isinstance(scores, torch.Tensor) and scores.ndim == 1
    
    if box.shape[-1] != 4:
        raise ValueError(
            f":param:`box` must have the shape of [N, 4]. But got: {box.shape}."
        )
    if box.shape[0] != scores.shape[0]:
        raise ValueError(
            f":param:`box` and :param:`scores` must have same length. "
            f"But got: {box.shape[0]} != {scores.shape[0]}."
        )

    x1, y1, x2, y2 = box.unbind(-1)
    areas 	       = (x2 - x1) * (y2 - y1)
    _, order       = scores.sort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i     = order[0]
        keep.append(i)
        xx1   = torch.max(x1[i], x1[order[1:]])
        yy1   = torch.max(y1[i], y1[order[1:]])
        xx2   = torch.min(x2[i], x2[order[1:]])
        yy2   = torch.min(y2[i], y2[order[1:]])
        w     = torch.clamp(xx2 - xx1, min=0.)
        h     = torch.clamp(yy2 - yy1, min=0.)
        inter = w * h
        ovr   = inter / (areas[i] + areas[order[1:]] - inter)
        inds  = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    if len(keep) > 0:
        return torch.stack(keep)
    return torch.tensor(keep)

# endregion
