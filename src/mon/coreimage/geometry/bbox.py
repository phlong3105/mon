#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements geometry functions for bounding boxes."""

from __future__ import annotations

__all__ = [
    "affine_bbox", "bbox_cxcyar_to_cxcyrh", "bbox_cxcyar_to_cxcywh",
    "bbox_cxcyar_to_cxcywhn", "bbox_cxcyar_to_xywh", "bbox_cxcyar_to_xyxy",
    "bbox_cxcyrh_to_cxcyar", "bbox_cxcyrh_to_cxcywh", "bbox_cxcyrh_to_cxcywhn",
    "bbox_cxcyrh_to_xywh", "bbox_cxcyrh_to_xyxy", "bbox_cxcywh_to_cxcyar",
    "bbox_cxcywh_to_cxcyrh", "bbox_cxcywh_to_cxcywhn", "bbox_cxcywh_to_xywh",
    "bbox_cxcywh_to_xyxy", "bbox_cxcywhn_to_cxcyar", "bbox_cxcywhn_to_cxcyrh",
    "bbox_cxcywhn_to_cxcywh", "bbox_cxcywhn_to_xywh", "bbox_cxcywhn_to_xyxy",
    "bbox_xywh_to_cxcyar", "bbox_xywh_to_cxcyrh", "bbox_xywh_to_cxcywh",
    "bbox_xywh_to_cxcywhn", "bbox_xywh_to_xyxy", "bbox_xyxy_to_cxcyar",
    "bbox_xyxy_to_cxcyrh", "bbox_xyxy_to_cxcywh", "bbox_xyxy_to_cxcywhn",
    "bbox_xyxy_to_xywh", "clip_bbox", "flip_bbox_horizontal",
    "flip_bbox_vertical", "generate_bbox", "get_bbox_area", "get_bbox_center",
    "get_bbox_corners", "get_bbox_corners_points",
    "get_bbox_intersection_union", "get_bbox_iou", "get_enclosing_bbox",
    "get_single_bbox_iou", "nms", "rotate_bbox", "scale_bbox",
    "scale_bbox_original", "shear_bbox", "translate_bbox",
    "translate_bbox_horizontal", "translate_bbox_vertical",
]

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from mon import core
from mon.coreimage import util

if TYPE_CHECKING:
    from mon.coreimage.typing import Floats, Ints, TensorOrArray
    

# region Affine Transform

def affine_bbox(
    bbox      : torch.Tensor,
    image_size: Ints,
    angle     : float | int,
    translate : Ints,
    scale     : float | int,
    shear     : Floats,
    center    : Ints  | None = None,
    drop_ratio: float        = 0.0,
) -> torch.Tensor:
    """Apply an affine transformation on the bounding bbox.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using
        -opencv-in-python
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
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
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        A transformed bbox of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
    
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
    m      = torch.from_numpy(m).to(torch.double).to(bbox.device)
    
    bbox      = bbox.clone()
    n         = len(bbox)
    xy        = torch.ones((n * 4, 3), dtype=bbox.dtype).to(bbox.devices)
    xy[:, :2] = bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy        = xy @ m.T  # Transform
    xy        = xy[:, :2].reshape(n, 8)
    
    x         = xy[:, [0, 2, 4, 6]]
    y         = xy[:, [1, 3, 5, 7]]
    x1        = torch.min(x, 1, keepdim=True).values
    y1        = torch.min(y, 1, keepdim=True).values
    x2        = torch.max(x, 1, keepdim=True).values
    y2        = torch.max(y, 1, keepdim=True).values
    xy        = torch.cat((x1, y1, x2, y2)).reshape(4, n).T
    bbox       = clip_bbox(
        bbox       = xy,
        image_size = image_size,
        drop_ratio = drop_ratio,
    )
    return bbox


def clip_bbox(
    bbox      : torch.Tensor,
    image_size: Ints,
    drop_ratio: float = 0.0,
) -> torch.Tensor:
    """Clip bounding boxes to an image size and removes the bounding boxes,
    which lose too much area as a result of the augmentation.
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: An image size of shape [H, W].
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Clipped bounding boxes of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
    h, w = util.to_size(image_size)
    area = get_bbox_area(bbox)

    bbox = bbox.clone()
    bbox[:, 0].clamp_(0, w)  # x1
    bbox[:, 1].clamp_(0, h)  # y1
    bbox[:, 2].clamp_(0, w)  # x2
    bbox[:, 3].clamp_(0, h)  # y2
    delta_area = ((area - get_bbox_area(bbox)) / area)
    mask       = (delta_area < (1 - drop_ratio)).to(torch.int)
    bbox       = bbox[mask == 1, :]
    return bbox


def flip_bbox_horizontal(
    bbox        : torch.Tensor,
    image_center: torch.Tensor
) -> torch.Tensor:
    """Horizontally flip boxes, which are specified by their normalized
    [cx, cy, w, h] coordinates.
    
    Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        bbox: Bounding boxes of shape [N, 4] to be flipped.
        image_center: The center of the image.
        
    Returns:
        Flipped bounding boxes of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
    assert isinstance(image_center, torch.Tensor) and image_center.ndim == 1
    bbox             = bbox.clone()
    bbox[:, [0, 2]] += 2 * (image_center[[0, 2]] - bbox[:, [0, 2]])
    box_w            = abs(bbox[:, 0] - bbox[:, 2])
    bbox[:, 0]      -= box_w
    bbox[:, 2]      += box_w
    return bbox


def flip_bbox_vertical(
    bbox        : torch.Tensor,
    image_center: torch.Tensor,
) -> torch.Tensor:
    """Flip bounding boxes vertically, which are specified by their [cx, cy, w,
    h] norm coordinates.
	
	Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        bbox: Bounding boxes of shape [N, 4] to be flipped.
        image_center: The center of the image.
        
    Returns:
        Flipped boxes of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
    bbox             = bbox.clone()
    bbox[:, [1, 3]] += 2 * (image_center[[0, 2]] - bbox[:, [1, 3]])
    box_h            = abs(bbox[:, 1] - bbox[:, 3])
    bbox[:, 1]      -= box_h
    bbox[:, 3]      += box_h
    return bbox


def rotate_bbox(
    bbox      : torch.Tensor,
    image_size: Ints,
    angle     : float,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Rotate the bounding bbox by the given magnitude.

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: The original image size.
        angle: An angle to rotate the bounding bbox.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
            
    Returns:
        A translated boxes of shape [N, 4].
    """
    bbox = affine_bbox(
        bbox= bbox,
        image_size = image_size,
        angle      = angle,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return bbox


def scale_bbox(
    bbox      : torch.Tensor,
    cur_size  : Ints,
    new_size  : Ints   | None = None,
    factor    : Floats | None = (1.0, 1.0),
    keep_shape: bool          = False,
    drop_ratio: float         = 0.0,
) -> torch.Tensor:
    """Scale bounding boxes coordinates by the given factor or by inferring
    from current image size and new size.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling
        -translation/

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        cur_size: The current image size.
        new_size: A new image size. Defaults to None.
        factor: A desired scaling factor in each direction. If scalar, the value
            is used for both the vertical and horizontal direction. Defaults to
            (1.0, 1.0).
        keep_shape: When True, translate the scaled bounding boxes. Defaults to
            False.
        drop_ratio: If a fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
            
    Returns:
        A scaled bounding boxes of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
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
    
    bbox         = bbox.clone()
    bbox[:, :4] *= [factor_hor, factor_ver, factor_hor, factor_ver]
    bbox         = clip_bbox(
        bbox=bbox, image_size=(h1, w1), drop_ratio=drop_ratio)
    if keep_shape and (h0 * w0) >= (h1 * w1):
        hor  = int(abs(w0 - w1) / 2)
        ver  = int(abs(h0 - h1) / 2)
        bbox = translate_bbox(
            bbox= bbox,
            magnitude  = (hor, ver),
            image_size = (h1, w1),
        )
    return bbox


def scale_bbox_original(
    bbox      : torch.Tensor,
    cur_size  : Ints,
    new_size  : Ints,
    ratio_pad = None,
) -> torch.Tensor:
    """Scale bounding boxes coordinates (from the :param:`cur_size`) to the
    :param:`new_size`.

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        cur_size: The current size.
        new_size: A new size.
        ratio_pad: Defaults to None.
        
    Returns:
        Scaled bounding boxes of shape [N, 4].
    """
    assert isinstance(bbox, torch.Tensor) and bbox.ndim == 2
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

    bbox             = bbox.clone()
    bbox[:, [0, 2]] -= pad[0]  # x padding
    bbox[:, [1, 3]] -= pad[1]  # y padding
    bbox[:, :4]     /= gain
    bbox             = clip_bbox(bbox=bbox, image_size=new_size)
    return bbox


def shear_bbox(
    bbox      : torch.Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Shear bounding boxes.
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: The original image size.
        magnitude: A shear magnitude.
        center: The center of affine transformation. If None, use the center
        of the
            image. Defaults to None.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Sheared bounding boxes of shape [N, 4].
    """
    bbox = affine_bbox(
        bbox       = bbox,
        image_size = image_size,
        angle      = 0.0,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = magnitude,
        center     = center,
        drop_ratio = drop_ratio,
    )
    return bbox


def translate_bbox(
    bbox      : torch.Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes.

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: The original image size.
        magnitude: A translation magnitude.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    bbox = affine_bbox(
        bbox       = bbox,
        image_size = image_size,
        angle      = 0.0,
        translate  = magnitude,
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return bbox
    

def translate_bbox_horizontal(
    bbox      : torch.Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes horizontally.

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: The original image size.
        magnitude: A horizontal translation magnitude.
        center: The center of affine transformation. If None, use the center
        of the
            image. Defaults to None.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    bbox = translate_bbox(
        bbox       = bbox,
        image_size = image_size,
        magnitude  = [magnitude, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return bbox


def translate_bbox_vertical(
    bbox      : torch.Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
) -> torch.Tensor:
    """Translate bounding boxes in vertical direction.

    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
        image_size: The original image size.
        magnitude: A vertical translation.
        center: The center of affine transformation. If None, use the center of
            the image. Defaults to None.
        drop_ratio: If the fraction of a bounding bbox left in the image after
            being clipped is less than :param:`drop_ratio` the bounding bbox is
            dropped. If :param:`drop_ratio` == 0, don't drop any bounding boxes.
            Defaults to 0.0.
        
    Returns:
        Translated boxes of shape [N, 4].
    """
    bbox = translate_bbox(
        bbox       = bbox,
        image_size = image_size,
        magnitude  = [0, magnitude],
        center     = center,
        drop_ratio = drop_ratio,
    )
    return bbox

# endregion


# region Box Format Conversion

def bbox_cxcyar_to_cxcyrh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, a, r] format to [cx, cy, r, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, a, r] format.
        
    Returns:
        Bounding boxes in [cx, cy, r, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, a, r, *_ = bbox.T
    w = torch.sqrt(a * r)
    h = a / w
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )


def bbox_cxcyar_to_cxcywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, a, r] format to [cx, cy, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, a, r] format.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, a, r, *_ = bbox.T
    w = torch.sqrt(a * r)
    h = a / w
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyar_to_cxcywhn(
    bbox   : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, a, r] format to [cx, cy, w, h]
    norm format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to the normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, a, r] format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] norm format.
    """
    bbox = util.upcast(bbox)
    cx, cy, a, r, *_ = bbox.T
    w       = torch.sqrt(a * r)
    h       = (a / w)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w / width
    h_norm  = h / height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyar_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, a, r] format to [x, y, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, a, r] format.
        
    Returns:
        Bounding boxes in [x, y, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, a, r, *_ = bbox.T
    w = torch.sqrt(a * r)
    h = a / w
    x = cx - (w / 2.0)
    y = cy - (h / 2.0)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )

    
def bbox_cxcyar_to_xyxy(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, a, r] format to [x1, y1, x2, y2]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, a, r] format.
        
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, a, r, *_ = bbox.T
    w  = torch.sqrt(a * r)
    h  = a / w
    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyrh_to_cxcyar(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, r, h] format to [cx, cy, a, r]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, r, h] format.
        
    Returns:
        Bounding boxes in [cx, cy, a, r] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, r, h, *_ = bbox.T
    w = r * h
    a = w * h
    r = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyrh_to_cxcywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, r, h] format to [cx, cy, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, r, h] format.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, r, h, *_ = bbox.T
    w    = r * h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyrh_to_cxcywhn(
    bbox   : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, r, h] format to [cx, cy, w, h]
    norm format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, r, h] format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
    """
    bbox    = util.upcast(bbox)
    cx, cy, r, h, *_ = bbox.T
    w       = r * h
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyrh_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, r, h] format to [x, y, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, r, h] format.
        
    Returns:
        Bounding boxes in [x, y, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, r, h, *_ = bbox.T
    w = r * h
    x = cx - w / 2.0
    y = cy - h / 2.0
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcyrh_to_xyxy(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, r, h] format to [x1, y1, x2, y2]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, r, h] format.
        
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, r, h, *_ = bbox.T
    w  = r * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywh_to_cxcyar(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] format to [cx, cy, a, r]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] format.
        
    Returns:
        Bounding boxes in [cx, cy, a, r] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, w, h, *_ = bbox.T
    a = w * h
    r = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywh_to_cxcyrh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] format to [cx, cy, r, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] format.
        
    Returns:
        Bounding boxes in [cx, cy, r, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, w, h, *_ = bbox.T
    r    = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywh_to_cxcywhn(
    bbox   : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] format to [cx, cy, r, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, r, h] norm format.
    """
    bbox    = util.upcast(bbox)
    cx, cy, w, h, *_ = bbox.T
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
    

def bbox_cxcywh_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] format to [x, y, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] format.
        
    Returns:
        Bounding boxes in [x, y, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, w, h, *_ = bbox.T
    x = cx - w / 2.0
    y = cy - h / 2.0
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
    

def bbox_cxcywh_to_xyxy(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] format to [x1, y1, x2, y2]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] format.
        
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox = util.upcast(bbox)
    cx, cy, w, h, *_ = bbox.T
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywhn_to_cxcyar(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] norm format to [cx, cy, a, r]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, a, r] format.
    """
    bbox = util.upcast(bbox)
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    cx = cx_norm * width
    cy = cy_norm * height
    a  = (w_norm * width) * (h_norm * height)
    r  = (w_norm * width) / (h_norm * height)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywhn_to_cxcyrh(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] norm format to [cx, cy, r, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, r, h] format.
    """
    bbox = util.upcast(bbox)
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    cx = cx_norm * width
    cy = cy_norm * height
    r  = (w_norm * width) / (h_norm * height)
    h  = h_norm * height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywhn_to_cxcywh(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] norm format to [cx, cy, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    cx = cx_norm * width
    cy = cy_norm * height
    w  = w_norm  * width
    h  = h_norm  * height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_cxcywhn_to_xywh(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] norm format to [x, y, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [x, y, w, h] format.
    """
    bbox = util.upcast(bbox)
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    w = w_norm * width
    h = h_norm * height
    x = (cx_norm * width) - (w / 2.0)
    y = (cy_norm * height) - (h / 2.0)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x, y, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x, y, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
   
   
def bbox_cxcywhn_to_xyxy(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [cx, cy, w, h] norm format to [x1, y1, x2,
    y2]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [cx, cy, w, h] norm format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox = util.upcast(bbox)
    cx_norm, cy_norm, w_norm, h_norm, *_ = bbox.T
    x1 = width  * (cx_norm - w_norm / 2)
    y1 = height * (cy_norm - h_norm / 2)
    x2 = width  * (cx_norm + w_norm / 2)
    y2 = height * (cy_norm + h_norm / 2)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, x2, y2), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x1, y1, x2, y2), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )


def bbox_xywh_to_cxcyar(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x, y, w, h] format to [cx, cy, a, r] format.
    
    [cx, cy] refers to the center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x, y, w, h] format.
       
    Returns:
        Bounding boxes in [cx, cy, a, r] format.
    """
    bbox = util.upcast(bbox)
    x, y, w, h, *_ = bbox.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    a  = w * h
    r  = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xywh_to_cxcyrh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x, y, w, h] format to [cx, cy, r, h] format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x, y, w, h] format.
       
    Returns:
        Bounding boxes in [cx, cy, r, h] format.
    """
    bbox = util.upcast(bbox)
    x, y, w, h, *_ = bbox.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    r  = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )
    

def bbox_xywh_to_cxcywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x, y, w, h] format to [cx, cy, w, h] format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x, y, w, h] format.
       
    Returns:
        Bounding boxes in [cx, cy, w, h] format.
    """
    bbox = util.upcast(bbox)
    x, y, w, h, *_ = bbox.T
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xywh_to_cxcywhn(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [x, y, w, h] format to [cx, cy, w, h] norm
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to the normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [x, y, w, h] format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] norm format.
    """
    bbox = util.upcast(bbox)
    x, y, w, h, *_ = bbox.T
    cx      = x + (w / 2.0)
    cy      = y + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xywh_to_xyxy(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x, y, w, h] format to [x1, y1, x2, y2]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x, y, w, h] format.
       
    Returns:
        Bounding boxes in [x1, y1, x2, y2] format.
    """
    bbox = util.upcast(bbox)
    x, y, w, h, *_ = bbox.T
    x2 = x + w
    y2 = y + h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x, y, x2, y2), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x, y, x2, y2), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xyxy_to_cxcyar(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [cx, cy, a, r]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x1, y1, x2, y2] format.
       
    Returns:
        Bounding boxes in [cx, cy, a, r] format.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    a  = w * h
    r  = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, a, r), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, a, r), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
    

def bbox_xyxy_to_cxcyrh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [cx, cy, r, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x1, y1, x2, y2] format.
       
    Returns:
        Bounding boxes in [cx, cy, r, h] format.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    r  = w / h
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, r, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, r, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xyxy_to_cxcywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [cx, cy, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x1, y1, x2, y2] format.
       
    Returns:
        Bounding boxes in [cx, cy, w, h] format.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    w  = x2 - x1
    h  = y2 - y1
    cx = x1 + (w / 2.0)
    cy = y1 + (h / 2.0)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xyxy_to_cxcywhn(
    bbox  : TensorOrArray,
    height: int,
    width : int
) -> TensorOrArray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [cx, cy, w, h]
    norm format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
    _norm refers to the normalized value in the range `[0.0, 1.0]`. For example:
        `x_norm = absolute_x / image_width`
        `height_norm = absolute_height / image_height`.
    
    Args:
        bbox: Bounding boxes in [x1, y1, x2, y2] format.
        height: The height of the image.
        width: The width of the image.
        
    Returns:
        Bounding boxes in [cx, cy, w, h] norm format.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    w       = x2 - x1
    h       = y2 - y1
    cx      = x1 + (w / 2.0)
    cy      = y1 + (h / 2.0)
    cx_norm = cx / width
    cy_norm = cy / height
    w_norm  = w  / width
    h_norm  = h  / height
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx_norm, cy_norm, w_norm, h_norm), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )


def bbox_xyxy_to_xywh(bbox: TensorOrArray) -> TensorOrArray:
    """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, w, h]
    format.
    
    [cx, cy] refers to center of bounding bbox.
    [a, r] refers to area (width * height) and aspect ratio (width / height) of
        bounding bbox.
    [w, h] refers to width and height of bounding bbox.
   
    Args:
        bbox: Bounding boxes in [x1, y1, x2, y2] format.
       
    Returns:
        Bounding boxes in [x, y, w, h] format.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    w = x2 - x1
    h = y2 - y1
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, w, h), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((x1, y1, w, h), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )

# endregion


# region Box Property

def get_bbox_area(bbox: TensorOrArray) -> TensorOrArray:
    """Compute the area of bounding bbox(es), which are specified by their
    [x1, y1, x2, y2] coordinates.
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
    
    Returns:
        The area for each bbox.
    """
    bbox = util.upcast(bbox)
    x1, y1, x2, y2, *_ = bbox.T
    area = (x2 - x1) * (y2 - y1)
    return area


def get_bbox_center(bbox: TensorOrArray) -> TensorOrArray:
    """Compute the center of bounding bbox(es), which are specified by their
    [x1, y1, x2, y2] coordinates.
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
    
    Returns:
        The center for each bbox of shape [N, 2].
    """
    bbox = util.upcast(bbox)
    bbox = bbox_xyxy_to_cxcywh(bbox)
    cx, cy, w, h = bbox.T
    if isinstance(bbox, torch.Tensor):
        return torch.stack((cx, cy), -1)
    elif isinstance(bbox, np.ndarray):
        return np.stack((cx, cy), -1)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )
    

def get_bbox_corners(bbox: TensorOrArray) -> TensorOrArray:
    """Get corners of bounding boxes.
    
    Args:
        bbox: Bounding boxes of shape [N, 4] and in [x1, y1, x2, y2] format.
    
    Returns:
        A tensor of shape `N x 8` containing N bounding boxes each described by
        their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).
    """
    width   = (bbox[:, 2] - bbox[:, 0]).reshape(-1, 1)
    height  = (bbox[:, 3] - bbox[:, 1]).reshape(-1, 1)
    x1      = bbox[:, 0].reshape(-1, 1)
    y1      = bbox[:, 1].reshape(-1, 1)
    x2      = x1 + width
    y2      = y1
    x3      = x1
    y3      = y1 + height
    x4      = bbox[:, 2].reshape(-1, 1)
    y4      = bbox[:, 3].reshape(-1, 1)
    if isinstance(bbox, torch.Tensor):
        return torch.stack((x1, y1, x2, y2, x3, y3, x4, y4))
    elif isinstance(bbox, np.ndarray):
        return np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )


def get_bbox_corners_points(bbox: TensorOrArray) -> TensorOrArray:
    """Get corners of bounding boxes as points.
    
    Args:
        bbox: Bounding boxes of shape [N, 4]. They are expected to be in (x1,
        y1,
            x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        Corners.
    """
    if bbox.ndim == 2:
        width  = (bbox[:, 2] - bbox[:, 0]).reshape(-1, 1)
        height = (bbox[:, 3] - bbox[:, 1]).reshape(-1, 1)
        x1     = bbox[:, 0].reshape(-1, 1)
        y1     = bbox[:, 1].reshape(-1, 1)
        x2     = x1 + width
        y2     = y1
        x3     = x1
        y3     = y1 + height
        x4     = bbox[:, 2].reshape(-1, 1)
        y4     = bbox[:, 3].reshape(-1, 1)
    else:
        width  = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x1     = bbox[0]
        y1     = bbox[1]
        x2     = x1 + width
        y2     = y1
        x3     = x1
        y3     = y1 + height
        x4     = bbox[2]
        y4     = bbox[3]
    if isinstance(bbox, torch.Tensor):
        return torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    elif isinstance(bbox, np.ndarray):
        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or "
            f":class:`numpy.ndarray`."
        )
    

def get_bbox_intersection_union(
    bbox1: TensorOrArray,
    bbox2: TensorOrArray
) -> tuple[TensorOrArray, TensorOrArray]:
    """Compute the intersection and union of two sets of boxes.
    
    References:
        https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    
    Args:
        bbox1: The first set of boxes of shape [N, 4] and in [x1, y1, x2, y2]
            format.
        bbox2: The second set of boxes of shape [N, 4] and in [x1, y1, x2, y2]
            format.
        
    Returns:
        Intersection.
        Union.
    """
    assert bbox1.ndim == 2 and bbox2.ndim == 2
    assert type(bbox1) == type(bbox2)
    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    if isinstance(bbox1, torch.Tensor):
        lt = torch.max(bbox1[:, None, :2], bbox2[:, :2])  # [N, M, 2]
        rb = torch.min(bbox1[:, None, 2:], bbox2[:, 2:])  # [N, M, 2]
    elif isinstance(bbox1, np.ndarray):
        lt = np.max(bbox1[:, None, :2], bbox2[:, :2])  # [N, M, 2]
        rb = np.min(bbox1[:, None, 2:], bbox2[:, 2:])  # [N, M, 2]
    wh    = util.upcast(rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter, union


def get_bbox_iou(
    bbox1: TensorOrArray,
    bbox2: TensorOrArray,
) -> TensorOrArray:
    """Return the intersection-over-union (Jaccard index) between two sets of
    boxes.
    
    Args:
        bbox1: The first set of boxes of shape [N, 4] and in [x1, y1, x2, y2]
            format.
        bbox2: The second set of boxes of shape [M, 4] and in [x1, y1, x2, y2]
            format.
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element
        in boxes1 and boxes2.
    """
    inter, union = get_bbox_intersection_union(bbox1=bbox1, bbox2=bbox2)
    iou          = inter / union
    return iou


def get_single_bbox_iou(
    bbox1: TensorOrArray,
    bbox2: TensorOrArray,
) -> TensorOrArray | float:
    """Return intersection-over-union (Jaccard index) between two boxes.
    
    Args:
        bbox1: The first bbox of shape [4] and in [x1, y1, x2, y2] format.
        bbox2: The second bbox of shape [4] and in [x1, y1, x2, y2] format.
    
    Returns:
        The IoU value for :param:`bbox1` and :param:`bbox2`.
    """
    if isinstance(bbox1, torch.Tensor) and type(bbox1) == type(bbox2):
        xx1  = torch.maximum(bbox1[0], bbox2[0])
        yy1  = torch.maximum(bbox1[1], bbox2[1])
        xx2  = torch.minimum(bbox1[2], bbox2[2])
        yy2  = torch.minimum(bbox1[3], bbox2[3])
        w    = torch.maximum(torch.Tensor(0.0), xx2 - xx1)
        h    = torch.maximum(torch.Tensor(0.0), yy2 - yy1)
    elif isinstance(bbox1, np.ndarray) and type(bbox1) == type(bbox2):
        xx1  = np.maximum(bbox1[0], bbox2[0])
        yy1  = np.maximum(bbox1[1], bbox2[1])
        xx2  = np.minimum(bbox1[2], bbox2[2])
        yy2  = np.minimum(bbox1[3], bbox2[3])
        w    = np.maximum(0.0, xx2 - xx1)
        h    = np.maximum(0.0, yy2 - yy1)
    else:
        raise TypeError
    wh  = w * h
    iou = wh / ((bbox1[2] - bbox1[0]) *
                (bbox1[3] - bbox1[1]) +
                (bbox2[2] - bbox2[0]) *
                (bbox2[3] - bbox2[1]) - wh)
    return iou


def get_enclosing_bbox(bbox: TensorOrArray) -> TensorOrArray:
    """Get an enclosing bbox for rotated corners of a bounding bbox.
    
    Args:
        bbox: Bounding of shape [N, 8], containing N bounding boxes each
            described by their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Returns:
        Bounding boxes of shape [N, 4]. They are expected to be in [x1, y1, x2,
        y2] and in [x1, y1, x2, y2] format.
    """
    x_    = bbox[:, [0, 2, 4, 6]]
    y_    = bbox[:, [1, 3, 5, 7]]
    if isinstance(bbox, torch.Tensor):
        x1    = torch.min(x_, 1).reshape(-1, 1)
        y1    = torch.min(y_, 1).reshape(-1, 1)
        x2    = torch.max(x_, 1).reshape(-1, 1)
        y2    = torch.max(y_, 1).reshape(-1, 1)
        return torch.stack((x1, y1, x2, y2, bbox[:, 8:]))
    elif isinstance(bbox, np.ndarray):
        x1    = np.min(x_, 1).reshape(-1, 1)
        y1    = np.min(y_, 1).reshape(-1, 1)
        x2    = np.max(x_, 1).reshape(-1, 1)
        y2    = np.max(y_, 1).reshape(-1, 1)
        return np.hstack((x1, y1, x2, y2, bbox[:, 8:]))
    else:
        raise TypeError(
            f":param:`bbox` must be a :class`torch.Tensor` or :class:`numpy.ndarray`."
        )


def generate_bbox(
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
        Bounding bbox.

    Examples:
        >>> x_start = torch.Tensor([0, 1])
        >>> y_start = torch.Tensor([1, 0])
        >>> width   = torch.Tensor([5, 3])
        >>> height  = torch.Tensor([7, 4])
        >>> generate_bbox(x_start, y_start, width, height)
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


def nms(
    box          : torch.Tensor,
    scores       : torch.Tensor,
    iou_threshold: float
) -> torch.Tensor:
    """Performs non-maxima suppression (NMS) on a given image of bounding boxes
    according to the intersection-over-union (IoU).
    
    NMS iteratively removes lower scoring boxes which have an IoU greater than
    `iou_threshold` with another (higher scoring) bbox.
    
    If multiple boxes have the exact same score and satisfy the IoU criterion
    with respect to a reference bbox, the selected bbox is not guaranteed to be
    the same between CPU and GPU. This is similar to the behavior of argsort in
    PyTorch when repeated values are present.
    
    Args:
        box: Bounding boxes of shape [N, 4] to perform NMS on. They are expected
            to be in [x1, y1, x2, y2] format with `0 <= x1 < x2` and
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
        >>> nms(bbox, scores, iou_threshold=0.8)
        image([0, 3, 1])
    """
    assert isinstance(box, torch.Tensor)    and box.ndim == 2
    assert isinstance(scores, torch.Tensor) and scores.ndim == 1
    
    if box.shape[-1] != 4:
        raise ValueError(
            f":param:`bbox` must have the shape of [N, 4]. But got: {box.shape}."
        )
    if box.shape[0] != scores.shape[0]:
        raise ValueError(
            f":param:`bbox` and :param:`scores` must have same length. "
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
