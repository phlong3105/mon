#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for bounding box. For example: format conversion, geometric
calculations, box metrics, ...
"""

from __future__ import annotations

import math
import random
from typing import Optional
from typing import Union

import cv2
import numpy as np
import torch
from torch import Tensor

from one.core import Float2T
from one.core import FloatAnyT
from one.core import Int2Or3T
from one.core import Int2T
from one.core import IntAnyT
from one.core import ListOrTuple2T
from one.core import TensorOrArray
from one.core import to_size
from one.core import upcast
from one.vision.shape.box_convert import box_xyxy_to_cxcywh

__all__ = [
    "affine_box",
    "clip_box",
    "compute_box_area",
    "compute_box_intersection",
    "compute_box_intersection_union",
    "compute_box_iou",
    "compute_box_iou_old",
    "compute_single_box_iou",
    "cutout_box",
    "generate_box",
    "get_box_center",
    "get_box_corners",
    "get_box_corners_points",
    "get_enclosing_box",
    "hflip_box",
    "htranslate_box",
    "is_box_candidates",
    "nms",
    "rotate_box",
    "scale_box",
    "scale_box_original",
    "shear_box",
    "translate_box",
    "vflip_box",
    "vtranslate_box",
]


# MARK: - Functional

def _affine_tensor_box(
    box       : Tensor,
    image_size: Int2Or3T,
    angle     : float,
    translate : IntAnyT,
    scale     : float,
    shear     : FloatAnyT,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
    
    Args:
        box (Tensor[N, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W]):
            Image size.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale.
        shear (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (Tensor[N, 4]):
            Transformed box.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(f"`angle` must be `int` or `float`. But got: {type(angle)}.")
    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if not isinstance(translate, (list, tuple)):
        raise TypeError(f"`translate` must be `list` or `tuple`. But got: {type(translate)}.")
    if isinstance(translate, tuple):
        translate = list(translate)
    if len(translate) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(translate)}.")
    
    if isinstance(scale, int):
        scale = float(scale)
    if scale < 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")
   
    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(f"`shear` must be a single value or a sequence of length 2. But got: {shear}.")
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(shear)}.")
        
    h, w   = to_size(image_size)
    center = (h * 0.5, w * 0.5) if center is None else center
    center = tuple(center[::-1])
    angle  = -angle
    R      = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    T      = translate
    S      = [math.radians(-shear[0]), math.radians(-shear[1])]
    M      = np.float32([[R[0, 0]       , S[0] + R[0, 1], R[0, 2] + T[0] + (-S[0] * center[1])],
                         [S[1] + R[1, 0], R[1, 1]       , R[1, 2] + T[1] + (-S[1] * center[0])],
                         [0             , 0             , 1]])
    M      = torch.from_numpy(M).to(torch.double).to(box.device)

    # NOTE: Create new boxes
    n         = len(box)
    xy        = torch.ones((n * 4, 3), dtype=box.dtype).to(box.device)
    xy[:, :2] = box[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy        = xy @ M.T  # Transform
    xy        = xy[:, :2].reshape(n, 8)
    
    x         = xy[:, [0, 2, 4, 6]]
    y         = xy[:, [1, 3, 5, 7]]
    x1        = torch.min(x, 1, keepdim=True).values
    y1        = torch.min(y, 1, keepdim=True).values
    x2        = torch.max(x, 1, keepdim=True).values
    y2        = torch.max(y, 1, keepdim=True).values
    xy        = torch.cat((x1, y1, x2, y2)).reshape(4, n).T
    return clip_box(box=xy, image_size=image_size, drop_ratio=drop_ratio)


def _affine_numpy_box(
    box       : np.ndarray,
    image_size: Int2Or3T,
    angle     : float,
    translate : IntAnyT,
    scale     : float,
    shear     : FloatAnyT,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> np.ndarray:
    """Apply affine transformation on the image keeping image center invariant.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
    
    Args:
        box (np.ndarray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W]):
            Image size.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale.
        shear (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (np.ndarray[B, 4]):
            Transformed box.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(f"`angle` must be `int` or `float`. But got: {type(angle)}.")
    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if not isinstance(translate, (list, tuple)):
        raise TypeError(f"`translate` must be `list` or `tuple`. But got: {type(translate)}.")
    if isinstance(translate, tuple):
        translate = list(translate)
    if len(translate) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(translate)}.")
    
    if isinstance(scale, int):
        scale = float(scale)
    if scale < 0.0:
        raise ValueError(f"`scale` must be positive. But got: {scale}.")
   
    if not isinstance(shear, (int, float, list, tuple)):
        raise TypeError(f"`shear` must be a single value or a sequence of length 2. But got: {shear}.")
    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(f"`translate` must be a sequence of length 2. But got: {len(shear)}.")
        
    h, w   = to_size(image_size)
    center = (h * 0.5, w * 0.5) if center is None else center
    center = tuple(center[::-1])
    angle  = -angle
    R      = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    T      = translate
    S      = [math.radians(-shear[0]), math.radians(-shear[1])]
    M      = np.float32([[R[0, 0]       , S[0] + R[0, 1], R[0, 2] + T[0] + (-S[0] * center[1])],
                         [S[1] + R[1, 0], R[1, 1]       , R[1, 2] + T[1] + (-S[1] * center[0])],
                         [0             , 0             , 1]])
    
    # NOTE: Create new boxes
    n         = len(box)
    xy        = np.ones((n * 4, 3))
    xy[:, :2] = box[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy        = xy @ M.T  # Transform
    xy        = xy[:, :2].reshape(n, 8)
    x         = xy[:, [0, 2, 4, 6]]
    y         = xy[:, [1, 3, 5, 7]]
    xy        = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    
    return clip_box(box=xy, image_size=image_size, drop_ratio=drop_ratio)


def affine_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    angle     : float,
    translate : IntAnyT,
    scale     : float,
    shear     : FloatAnyT,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Apply affine transformation on the image keeping image center invariant.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
    
    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W]):
            Image size.
        angle (float):
            Rotation angle in degrees between -180 and 180, clockwise direction.
        translate (IntAnyT):
            Horizontal and vertical translations (post-rotation translation).
        scale (float):
            Overall scale.
        shear (FloatAnyT):
            Shear angle value in degrees between -180 to 180, clockwise
            direction. If a sequence is specified, the first value corresponds
            to a shear parallel to the x-axis, while the second value
            corresponds to a shear parallel to the y-axis.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Transformed box.
    """
    if isinstance(box, Tensor):
        return _affine_tensor_box(
            box        = box,
            image_size = image_size,
            angle      = angle,
            translate  = translate,
            scale      = scale,
            shear      = shear,
            center     = center,
            drop_ratio = drop_ratio,
        )
    elif isinstance(box, np.ndarray):
        return _affine_numpy_box(
            box        = box,
            image_size = image_size,
            angle      = angle,
            translate  = translate,
            scale      = scale,
            shear      = shear,
            center     = center,
            drop_ratio = drop_ratio,
        )
    else:
        raise TypeError(f"Do not support: {type(box)}.")
    

def clip_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    drop_ratio: float = 0.0
) -> TensorOrArray:
    """Clip bounding boxes to image size [H, W] and removes the bounding boxes
    which lose too much area as a result of the augmentation. Both sets of boxes
    are expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
    `0 <= y1 < y2`.
    
    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes.
        image_size (Dim2T[H, W], optional):
            Image size.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.

    Returns:
        box (TensorOrArray[B, 4]):
            Clipped bounding boxes.
    """
    h, w = to_size(image_size)
    area = compute_box_area(box)

    if isinstance(box, Tensor):
        box = box.clone()
        box[:, 0].clamp_(0, w)  # x1
        box[:, 1].clamp_(0, h)  # y1
        box[:, 2].clamp_(0, w)  # x2
        box[:, 3].clamp_(0, h)  # y2
        delta_area = ((area - compute_box_area(box)) / area)
        mask       = (delta_area < (1 - drop_ratio)).to(torch.int)
        box        = box[mask == 1, :]
    elif isinstance(box, np.ndarray):
        box = box.copy()
        box[:, 0] = np.clip(box[:, 0], 0, image_size[1])  # x1
        box[:, 1] = np.clip(box[:, 1], 0, image_size[0])  # y1
        box[:, 2] = np.clip(box[:, 2], 0, image_size[1])  # x2
        box[:, 3] = np.clip(box[:, 3], 0, image_size[0])  # y2
        delta_area = ((area - compute_box_area(box)) / area)
        mask       = (delta_area < (1 - drop_ratio)).astype(int)
        box        = box[mask == 1, :]
    else:
        raise ValueError(f"Do not support {type(box)}.")

    return box


def compute_box_area(box: TensorOrArray) -> Union[TensorOrArray, float]:
    """Computes the area of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box (TensorOrArray[*, 4]):
            Boxes for which the area will be computed. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        (TensorOrArray[*, 4], float):
            The area for each box.
    """
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    return (x2 - x1) * (y2 - y1)


def compute_box_intersection(box1: Tensor, box2: Tensor) -> Tensor:
    """Find the intersection between 2 bounding boxes. Both sets of boxes are
    expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <=
    y1 < y2`.
    
    Args:
        box1 (TensorOrArray[B, 4]):
            First set of boxes.
        box2 (TensorOrArray[B, 4]):
            Second set of boxes.
    """
    n      = box1.size(0)
    A      = box1.size(1)
    B      = box2.size(1)
    max_xy = torch.min(box1[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box2[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box1[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box2[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter  = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def compute_box_intersection_union(
    box1: TensorOrArray, box2: TensorOrArray
) -> tuple[TensorOrArray, TensorOrArray]:
    """Compute the intersection and union of two set of boxes. Both sets of
    boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Implementation from https://github.com/kuangliu/torchcv/blob/master
    /torchcv/utils/box.py
    with slight modifications.
    
    Args:
        box1 (TensorOrArray[B, 4]):
            First set of boxes.
        box2 (TensorOrArray[B, 4]):
            Second set of boxes.
    """
    if isinstance(box1, Tensor):
        max = torch.max
        min = torch.min
    else:
        max = np.max
        min = np.min
    
    area1 = compute_box_area(box1)
    area2 = compute_box_area(box2)

    lt    = max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb    = min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    if isinstance(box1, Tensor):
        wh = upcast(rb - lt).clamp(min=0)  # [N, M, 2]
    else:
        wh = upcast(rb - lt).clip(min=0)  # [N, M, 2]
        
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter, union


def compute_box_iou(box1: TensorOrArray, box2: TensorOrArray) -> TensorOrArray:
    """Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Args:
        box1 (TensorOrArray[N, 4]):
            First set of boxes.
        box2 (TensorOrArray[M, 4]):
            Second set of boxes.
    
    Returns:
        iou (TensorOrArray[N, M]):
            The NxM matrix containing the pairwise IoU values for every element
            in boxes1 and boxes2.
    """
    inter, union = compute_box_intersection_union(box1, box2)
    iou          = inter / union
    return iou


def compute_box_iou_old(box1: TensorOrArray, box2: TensorOrArray) -> TensorOrArray:
    """From SORT: Computes IOU between two sets of boxes.
    
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.

    Args:
        box1 (TensorOrArray[N, 4]):
            First set of boxes.
        box2 (TensorOrArray[N, 4]):
            Second set of boxes.
    
    Returns:
        iou (TensorOrArray[N, M]):
            The NxM matrix containing the pairwise IoU values for every element
            in boxes1 and boxes2.
    """
    if isinstance(box1, Tensor):
        maximum = torch.maximum
        minimum = torch.minimum
        box1  = torch.unsqueeze(box1, 1)
        box2  = torch.unsqueeze(box2, 0)
    else:
        maximum = np.maximum
        minimum = np.minimum
        box1  = np.expand_dims(box1, 1)
        box2  = np.expand_dims(box2, 0)
        
    # boxes1 = np.expand_dims(boxes1, 1)
    # boxes2 = np.expand_dims(boxes2, 0)
    xx1 = maximum(box1[..., 0], box2[..., 0])
    yy1 = maximum(box1[..., 1], box2[..., 1])
    xx2 = minimum(box1[..., 2], box2[..., 2])
    yy2 = minimum(box1[..., 3], box2[..., 3])
    w   = maximum(0.0, xx2 - xx1)
    h   = maximum(0.0, yy2 - yy1)
    wh  = w * h
    iou = wh / ((box1[..., 2] - box1[..., 0]) *
                (box1[..., 3] - box1[..., 1]) +
                (box2[..., 2] - box2[..., 0]) *
                (box2[..., 3] - box2[..., 1]) - wh)
    return iou


def compute_single_box_iou(box1: Tensor, box2: Tensor) -> Union[TensorOrArray, float]:
    """Return intersection-over-union (Jaccard index) between two boxes.
    Both boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Args:
        box1 (TensorOrArray[4]):
            First box.
        box2 (TensorOrArray[4]):
            Second box.
    
    Returns:
        (TensorOrArray, float):
            IoU value for box1 and box2.
    """
    if isinstance(box1, Tensor):
        maximum = torch.maximum
        minimum = torch.minimum
    else:
        maximum = np.maximum
        minimum = np.minimum
        
    xx1 = maximum(box1[0], box2[0])
    yy1 = maximum(box1[1], box2[1])
    xx2 = minimum(box1[2], box2[2])
    yy2 = minimum(box1[3], box2[3])
    w   = maximum(0.0, xx2 - xx1)
    h   = maximum(0.0, yy2 - yy1)
    wh  = w * h
    ou  = wh / ((box1[2] - box1[0]) * (box1[3] - box1[1]) +
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - wh)
    return ou


def cutout_box(image: np.ndarray, box_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies image cutout augmentation with bounding box labels.

    References:
        https://arxiv.org/abs/1708.04552

    Args:
        image (np.ndarray):
            Image.
        box_labels (np.ndarray):
            Bounding box labels where the box coordinates are located at:
            labels[:, 2:6].

    Returns:
        image_cutout (np.ndarray):
            Cutout image.
        box_labels_cutout (np.ndarray):
            Cutout labels.
    """
    h, w              = image.shape[:2]
    image_cutout      = image.copy()
    box_labels_cutout = box_labels.copy()
    
    # NOTE: Create random masks
    scales = ([0.5] * 1 +
              [0.25] * 2 +
              [0.125] * 4 +
              [0.0625] * 8 +
              [0.03125] * 16)  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))
        
        # Box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        
        # Apply random color mask
        image_cutout[ymin:ymax, xmin:xmax] = [random.randint(64, 191)
                                              for _ in range(3)]
        
        # Return unobscured bounding boxes
        if len(box_labels_cutout) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], np.float32)
            # Intersection over area
            ioa = ioa(box, box_labels_cutout[:, 2:6])
            # Remove >60% obscured labels
            box_labels_cutout = box_labels_cutout[ioa < 0.60]
    
    return image_cutout, box_labels_cutout


def generate_box(x_start: Tensor, y_start: Tensor, width: Tensor, height: Tensor) -> Tensor:
    """Generate 2D bounding boxes according to the provided start coords,
    width and height.

    Args:
        x_start (Tensor):
            Tensor containing the x coordinates of the bounding boxes to be
            extracted. Shape must be a scalar image or [B].
        y_start (Tensor):
            Tensor containing the y coordinates of the bounding boxes to be
            extracted. Shape must be a scalar image or [B].
        width (Tensor):
            Widths of the masked image. Shape must be a scalar image or [B].
        height (Tensor):
            Heights of the masked image. Shape must be a scalar image or [B].

    Returns:
        box (Tensor):
            Bounding box image.

    Examples:
        >>> x_start = Tensor([0, 1])
        >>> y_start = Tensor([1, 0])
        >>> width   = Tensor([5, 3])
        >>> height  = Tensor([7, 4])
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


def get_box_center(box: TensorOrArray) -> TensorOrArray:
    """Computes the center of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box (TensorOrArray[*, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        center (Tensor[*, 2], np.ndarray[*, 2]):
            The center for each box.
    """
    box = upcast(box)
    box = box_xyxy_to_cxcywh(box)
    cx, cy, w, h = box.T
    
    if isinstance(box, Tensor):
        return torch.stack((cx, cy), -1)
    elif isinstance(box, np.ndarray):
        return np.stack((cx, cy), -1)
    else:
        raise ValueError(f"`box` must be a `Tensor` or `np.ndarray`. "
                         f"But got: {type(box)}.")


def get_box_corners(box: TensorOrArray) -> TensorOrArray:
    """Get corners of bounding boxes.
    
    Args:
        box (TensorOrArray[*, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        corners (Tensor[*, 8], np.ndarray[*, 8]):
            Shape `N x 8` containing N bounding boxes each described by their
            corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).
    """
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

    if isinstance(box, Tensor):
        return torch.stack((x1, y1, x2, y2, x3, y3, x4, y4))
    else:
        return np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))


def get_box_corners_points(box: TensorOrArray) -> TensorOrArray:
    """Get corners of bounding boxes as points.
    
    Args:
        box (TensorOrArray[*, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        corners (Tensor, np.ndarray):
    """
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
        
    if isinstance(box, Tensor):
        return torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    else:
        return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)


def get_enclosing_box(box: TensorOrArray) -> TensorOrArray:
    """Get an enclosing box for rotated corners of a bounding box.
    
    Args:
        box (TensorOrArray[*, 8]):
            Shape `N x 8` containing N bounding boxes each described by their
            corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Returns:
        box (TensorOrArray[*, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """
    min   = np.min    if isinstance(box, np.ndarray) else torch.min
    max   = np.max    if isinstance(box, np.ndarray) else torch.max
    stack = np.hstack if isinstance(box, np.ndarray) else torch.stack
    
    x_    = box[:, [0, 2, 4, 6]]
    y_    = box[:, [1, 3, 5, 7]]
    x1    = min(x_, 1).reshape(-1, 1)
    y1    = min(y_, 1).reshape(-1, 1)
    x2    = max(x_, 1).reshape(-1, 1)
    y2    = max(y_, 1).reshape(-1, 1)
    final = stack((x1, y1, x2, y2, box[:, 8:]))
    return final


def hflip_box(box: TensorOrArray, image_center: TensorOrArray) -> TensorOrArray:
    """Flip boxes horizontally, which are specified by their (cx, cy, w, h) norm
    coordinates.
    
    Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box (TensorOrArray[B, 4]):
            Boxes to be flipped.
        image_center (TensorOrArray[4]):
            Center of the image.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Flipped boxes.
    """
    box[:, [0, 2]] += 2 * (image_center[[0, 2]] - box[:, [0, 2]])
    box_w           = abs(box[:, 0] - box[:, 2])
    box[:, 0]      -= box_w
    box[:, 2]      += box_w
    return box


def htranslate_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    magnitude : int,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Translate the bounding box in horizontal direction.

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W, *]):
            Original image size.
        magnitude (int):
             Horizontally translation.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    '''
    if not isinstance(magnitude, (int, float)):
        raise TypeError(f"`magnitude` must be an `int` or `float`. But got: {type(magnitude)}.")
    return clip_box(
        box        = box[:, :4] + [magnitude, 0, magnitude, 0],
        image_size = image_size,
        drop_ratio = drop_ratio,
    )
    '''
    return translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [magnitude, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )


def is_box_candidates(
    box1    : np.ndarray,
    box2    : np.ndarray,
    wh_thr  : float = 2,
    ar_thr  : float = 20,
    area_thr: float = 0.2
) -> bool:
    """Return `True` if xyxy2 is the candidate for xyxy1.
    
    Args:
        box1 (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        box2 (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        wh_thr (float):
            Threshold of both width and height (pixels).
        ar_thr (float):
            Aspect ratio threshold.
        area_thr (float):
            Area ratio threshold.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar     = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # Aspect ratio
    return ((w2 > wh_thr) &
            (h2 > wh_thr) &
            (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) &
            (ar < ar_thr))  # candidates


def nms(box: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Perform non-maxima suppression (NMS) on a given image of bounding boxes
    according to the intersection-over-union (IoU).
    
    NMS iteratively removes lower scoring boxes which have an IoU greater than
    `iou_threshold` with another (higher scoring) box.
    
    If multiple boxes have the exact same score and satisfy the IoU criterion
    with respect to a reference box, the selected box is not guaranteed to be
    the same between CPU and GPU. This is similar to the behavior of argsort in
    PyTorch when repeated values are present.
    
    Args:
        box (Tensor[N, 4]):
            Boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2)
            format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        scores (Tensor[N]):
            Scores for each one of the boxes.
        iou_threshold (float):
            Discards all overlapping boxes with IoU > iou_threshold

    Return:
        (Tensor):
            Indices of the elements that have been kept by NMS, sorted in
            decreasing order of scores

    Example:
        >>> boxes  = Tensor([
        ...     [10., 10., 20., 20.],
        ...     [15., 5., 15., 25.],
        ...     [100., 100., 200., 200.],
        ...     [100., 100., 200., 200.]])
        >>> scores = Tensor([0.9, 0.8, 0.7, 0.9])
        >>> nms(box, scores, iou_threshold=0.8)
        image([0, 3, 1])
    """
    if box.ndim != 2 and box.shape[-1] != 4:
        raise ValueError(f"`box` must have the shape of [N, 4]. But got: {box.shape}.")
    if scores.ndim != 1:
        raise ValueError(f"`scores` must have the shape of [N]. But got: {scores.shape}.")
    if box.shape[0] != scores.shape[0]:
        raise ValueError(f"`box` and `scores` must have same shape. "
                         f"But got: {box.shape, scores.shape}.")

    x1, y1, x2, y2 = box.unbind(-1)
    areas 	       = (x2 - x1) * (y2 - y1)
    _, order       = scores.sort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i   = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w     = torch.clamp(xx2 - xx1, min=0.)
        h     = torch.clamp(yy2 - yy1, min=0.)
        inter = w * h
        ovr   = inter / (areas[i] + areas[order[1:]] - inter)

        inds  = torch.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    if len(keep) > 0:
        return torch.stack(keep)
    return torch.tensor(keep)


def rotate_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    angle     : float,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Rotate the bounding box by the given magnitude.

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W, *]):
            Original image size.
        angle (float):
			Angle to rotate the bounding box.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    return affine_box(
        box        = box,
        image_size = image_size,
        angle      = angle,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )


def scale_box(
    box       : TensorOrArray,
    cur_size  : Int2Or3T,
    new_size  : Optional[Int2Or3T] = None,
    factor    : Optional[Float2T]  = (1.0, 1.0),
    keep_shape: bool               = False,
    drop_ratio: float              = 0.0
) -> TensorOrArray:
    """Scale bounding boxes coordinates by the given factor or by inferring from
    current image size and new size.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size (Int2Or3T[H, W, *]):
            Current image size.
        new_size (Int2Or3T[W, H, *]):
            New image size. Default: `None`.
        factor (Float2T):
            Desired scaling factor in each direction. If scalar, the value is
            used for both the vertical and horizontal direction.
            Default: `(1.0, 1.0)`.
        keep_shape (bool):
            When `True`, translate the scaled bounding boxes. Default: `False`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Scaled bounding boxes.
    """
    h0, w0 = to_size(cur_size)  # H, W
    
    if new_size is not None:
        h1, w1 = to_size(new_size)  # H, W
        factor_ver, factor_hor = float(h1 / h0), float(w1 / w0)
    elif isinstance(factor, float):
        factor_ver = factor_hor = factor
        h1, w1 = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    else:
        factor_ver, factor_hor = factor
        h1, w1 = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    
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
    box       : TensorOrArray,
    cur_size  : Int2Or3T,
    new_size  : Int2Or3T,
    ratio_pad = None
) -> TensorOrArray:
    """Scale bounding boxes coordinates (from detector size) to the original
    image size.

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size (Int2Or3T):
            Detector's input size as [W, H, *].
        new_size (Int2Or3T):
            Original image size as [W, H, *].
        ratio_pad:

    Returns:
        box (TensorOrArray[B, 4]):
            Scaled bounding boxes.
    """
    cur_size = to_size(cur_size)
    new_size = to_size(new_size)

    if ratio_pad is None:  # Calculate from new_size
        gain = min(cur_size[0] / new_size[0],
                   cur_size[1] / new_size[1])  # gain  = old / new
        pad  = (cur_size[1] - new_size[1] * gain) / 2, \
               (cur_size[0] - new_size[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad  = ratio_pad[1]
    
    box[:, [0, 2]] -= pad[0]  # x padding
    box[:, [1, 3]] -= pad[1]  # y padding
    box[:, :4]     /= gain
    return clip_box(box, new_size)


def shear_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    magnitude : Int2T,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Shear bounding boxes coordinates by the given magnitude.
    
    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W, *]):
            Original image size.
        magnitude (Int2T[hor, ver]):
             Shear magnitude.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Sheared bounding boxes.
    """
    return affine_box(
        box        = box,
        image_size = image_size,
        angle      = 0.0,
        translate  = [0, 0],
        scale      = 1.0,
        shear      = magnitude,
        center     = center,
        drop_ratio = drop_ratio,
    )


def translate_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    magnitude : Int2T,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Translate the bounding box by the given magnitude.

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W, *]):
            Original image size.
        magnitude (Int2T[hor, ver]):
             Translation magnitude.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    '''
    if isinstance(magnitude, (int, float)):
        hor = ver = int(magnitude)
    else:
        hor, ver  = magnitude
    return clip_box(
        box        = box[:, :4] + [hor, ver, hor, ver],
        image_size = image_size,
        drop_ratio = drop_ratio,
    )
    '''
    return affine_box(
        box        = box,
        image_size = image_size,
        angle      = 0.0,
        translate  = magnitude,
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
    )
    

def vflip_box(box: TensorOrArray, image_center: TensorOrArray) -> TensorOrArray:
    """Flip boxes vertically, which are specified by their (cx, cy, w, h) norm
    coordinates.
	
	Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box (TensorOrArray[B, 4]):
            Boxes to be flipped.
        image_center (TensorOrArray[4]):
            Center of the image.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Flipped boxes.
    """
    box[:, [1, 3]] += 2 * (image_center[[0, 2]] - box[:, [1, 3]])
    box_h           = abs(box[:, 1] - box[:, 3])
    box[:, 1]      -= box_h
    box[:, 3]      += box_h
    return box


def vtranslate_box(
    box       : TensorOrArray,
    image_size: Int2Or3T,
    magnitude : int,
    center    : Optional[ListOrTuple2T[int]] = None,
    drop_ratio: float                        = 0.0,
) -> TensorOrArray:
    """Translate the bounding box in vertical direction.

    Args:
        box (TensorOrArray[B, 4]):
            Bounding boxes. They are expected to be in (x1, y1, x2, y2) format
            with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Int2Or3T[H, W, *]):
            Original image size.
        magnitude (int):
             Vertically translation.
        center (ListOrTuple2T[int], optional):
            Center of affine transformation.  If `None`, use the center of the
            image. Default: `None`.
        drop_ratio (float):
            If the fraction of a bounding box left in the image after being
            clipped is less than `drop_ratio` the bounding box is dropped.
            If `drop_ratio==0`, don't drop any bounding boxes. Default: `0.0`.
            
    Returns:
        box (TensorOrArray[B, 4]):
            Translated boxes.
    """
    '''
    if not isinstance(magnitude, (int, float)):
        raise TypeError(f"`magnitude` must be an `int` or `float`. But got: {type(magnitude)}.")
    return clip_box(
        box        = box[:, :4] + [0, magnitude, 0, magnitude],
        image_size = image_size,
        drop_ratio = drop_ratio,
    )
    '''
    return translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [0, magnitude],
        center     = center,
        drop_ratio = drop_ratio,
    )
