#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for bounding box. For example: format conversion, geometric
calculations, box metrics, ...
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torch import Tensor

from one.core import assert_number
from one.core import assert_positive_number
from one.core import assert_sequence_of_length
from one.core import assert_tensor
from one.core import assert_tensor_of_ndim
from one.core import Floats
from one.core import Ints
from one.core import to_size
from one.core import upcast
from one.vision.shape.box_convert import box_xyxy_to_cxcywh


# MARK: - Functional

def affine_box(
    box       : Tensor,
    image_size: Ints,
    angle     : float,
    translate : Ints,
    scale     : float,
    shear     : Floats,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
    inplace   : bool        = False,
) -> Tensor:
    """
    Apply affine transformation on the image keeping image center invariant.
    
    References:
        https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Image size of shape [H, W].
        angle (float): Rotation angle in degrees between -180 and 180,
            clockwise direction.
        translate (Ints): Horizontal and vertical translations (post-rotation
            translation).
        scale (float): Overall scale.
        shear (Floats): Shear angle value in degrees between -180 to 180,
            clockwise direction. If a sequence is specified, the first value
            corresponds to a shear parallel to the x-axis, while the second
            value corresponds to a shear parallel to the y-axis.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image 
            after being clipped is less than `drop_ratio` the bounding box is 
            dropped. If `drop_ratio==0`, don't drop any bounding boxes. 
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Transformed box of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    
    assert_number(angle)
    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, (int, float)):
        translate = [translate, translate]
    if isinstance(translate, tuple):
        translate = list(translate)
    assert_sequence_of_length(translate, 2)
    
    if isinstance(scale, int):
        scale = float(scale)
    assert_positive_number(scale)

    if isinstance(shear, (int, float)):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    assert_sequence_of_length(shear, 2)
        
    h, w   = to_size(image_size)
    center = (h * 0.5, w * 0.5) if center is None else center
    center = tuple(center[::-1])
    angle  = -angle
    r      = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    t      = translate
    s      = [math.radians(-shear[0]), math.radians(-shear[1])]
    m      = np.float32([[r[0, 0]       , s[0] + r[0, 1], r[0, 2] + t[0] + (-s[0] * center[1])],
                         [s[1] + r[1, 0], r[1, 1]       , r[1, 2] + t[1] + (-s[1] * center[0])],
                         [0             , 0             , 1]])
    m      = torch.from_numpy(m).to(torch.double).to(box.device)

    if not inplace:
        box = box.clone()
        
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
        inplace    = inplace,
    )
    return box


def clip_box(
    box       : Tensor,
    image_size: Ints,
    drop_ratio: float = 0.0,
    inplace   : bool  = False,
) -> Tensor:
    """
    Clip bounding boxes to image size [H, W] and removes the bounding boxes
    which lose too much area as a result of the augmentation. Both sets of boxes
    are expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
    `0 <= y1 < y2`.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4].
        image_size (Ints | None): Image size of shape [H, W].
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Clipped bounding boxes of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    h, w = to_size(image_size)
    area = compute_box_area(box)

    if not inplace:
        box = box.clone()

    box[:, 0].clamp_(0, w)  # x1
    box[:, 1].clamp_(0, h)  # y1
    box[:, 2].clamp_(0, w)  # x2
    box[:, 3].clamp_(0, h)  # y2
    delta_area = ((area - compute_box_area(box)) / area)
    mask       = (delta_area < (1 - drop_ratio)).to(torch.int)
    box        = box[mask == 1, :]
    return box


def compute_box_area(box: Tensor) -> Tensor:
    """
    Computes the area of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box (Tensor): Bounding boxes for which the area will be computed.
            They are expected to be in (x1, y1, x2, y2) format with
            `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        The area for each box.
    """
    assert_tensor_of_ndim(box, 2)
    box                = upcast(box)
    x1, y1, x2, y2, *_ = box.T
    return (x2 - x1) * (y2 - y1)


def compute_box_intersection_union(
    box1: Tensor, box2: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Compute the intersection and union of two set of boxes. Both sets of boxes
    are expected to be in (x1, y1, x2, y2) format with `0 <= x1 < x2` and
    `0 <= y1 < y2`.
    
    Implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    with slight modifications.
    
    Args:
        box1 (Tensor): First set of boxes of shape [N, 4].
        box2 (Tensor): Second set of boxes of shape [N, 4].
            
    Returns:
        Intersection.
        Union.
    """
    assert_tensor_of_ndim(box1, 2)
    assert_tensor_of_ndim(box2, 2)
    area1 = compute_box_area(box1)
    area2 = compute_box_area(box2)
    lt    = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb    = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    wh    = upcast(rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter
    return inter, union


def compute_box_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Args:
        box1 (Tensor): First set of boxes of shape [N, 4].
        box2 (Tensor): Second set of boxes of shape [M, 4].
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element
        in boxes1 and boxes2.
    """
    inter, union = compute_box_intersection_union(box1, box2)
    iou          = inter / union
    return iou


def compute_box_iou_old(box1: Tensor, box2: Tensor) -> Tensor:
    """
    From SORT: Computes IOU between two sets of boxes.
    
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.

    Args:
        box1 (Tensor): First set of boxes of shape [N, 4].
        box2 (Tensor): Second set of boxes of shape [M, 4].
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element in
        boxes1 and boxes2.
    """
    assert_tensor_of_ndim(box1, 2)
    assert_tensor_of_ndim(box2, 2)
    box1 = torch.unsqueeze(box1, 1)
    box2 = torch.unsqueeze(box2, 0)
    xx1  = torch.maximum(box1[..., 0], box2[..., 0])
    yy1  = torch.maximum(box1[..., 1], box2[..., 1])
    xx2  = torch.minimum(box1[..., 2], box2[..., 2])
    yy2  = torch.minimum(box1[..., 3], box2[..., 3])
    w    = torch.maximum(Tensor(0.0), xx2 - xx1)
    h    = torch.maximum(Tensor(0.0), yy2 - yy1)
    wh   = w * h
    iou  = wh / ((box1[..., 2] - box1[..., 0]) *
                 (box1[..., 3] - box1[..., 1]) +
                 (box2[..., 2] - box2[..., 0]) *
                 (box2[..., 3] - box2[..., 1]) - wh)
    return iou


def generate_box(
    x_start: Tensor, y_start: Tensor, width: Tensor, height: Tensor
) -> Tensor:
    """
    Generate 2D bounding boxes according to the provided start coords,
    width and height.

    Args:
        x_start (Tensor): Tensor containing the x coordinates of the bounding
            boxes to be extracted. Shape must be a scalar image or [B].
        y_start (Tensor): Tensor containing the y coordinates of the bounding
            boxes to be extracted. Shape must be a scalar image or [B].
        width (Tensor): Widths of the masked image. Shape must be a scalar
            image or [B].
        height (Tensor): Heights of the masked image. Shape must be a scalar
            image or [B].

    Returns:
        Bounding box.

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


def get_box_center(box: Tensor) -> Tensor:
    """
    Computes the center of bounding box(es), which are specified by their
    (x1, y1, x2, y2) coordinates.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        The center for each box of shape [N, 2].
    """
    assert_tensor_of_ndim(box, 2)
    box          = upcast(box)
    box          = box_xyxy_to_cxcywh(box)
    cx, cy, w, h = box.T
    return torch.stack((cx, cy), -1)


def get_box_corners(box: Tensor) -> Tensor:
    """
    Get corners of bounding boxes.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        Tensor of shape `N x 8` containing N bounding boxes each described by
        their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).
    """
    assert_tensor_of_ndim(box, 2)
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
    return torch.stack((x1, y1, x2, y2, x3, y3, x4, y4))


def get_box_corners_points(box: Tensor) -> Tensor:
    """
    Get corners of bounding boxes as points.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    
    Returns:
        corners (Tensor):
    """
    assert_tensor(box)
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
    return torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])


def get_enclosing_box(box: Tensor) -> Tensor:
    """
    Get an enclosing box for rotated corners of a bounding box.
    
    Args:
        box (Tensor): Bounding of shape [N, 8], containing N bounding boxes
            each described by their corner co-ordinates (x1 y1 x2 y2 x3 y3 x4 y4).

    Returns:
        Bounding boxes of shape [N, 4]. They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """
    assert_tensor_of_ndim(box, 2)
    x_    = box[:, [0, 2, 4, 6]]
    y_    = box[:, [1, 3, 5, 7]]
    x1    = torch.min(x_, 1).reshape(-1, 1)
    y1    = torch.min(y_, 1).reshape(-1, 1)
    x2    = torch.max(x_, 1).reshape(-1, 1)
    y2    = torch.max(y_, 1).reshape(-1, 1)
    final = torch.stack((x1, y1, x2, y2, box[:, 8:]))
    return final


def horizontal_flip_box(
    box: Tensor, image_center: Tensor, inplace: bool = False
) -> Tensor:
    """
    Horizontally flip boxes, which are specified by their (cx, cy, w, h) norm
    coordinates.
    
    Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box (Tensor): Bounding boxes of shape [N, 4] to be flipped.
        image_center (Tensor): Center of the image.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Flipped boxes of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    assert_tensor_of_ndim(image_center, 1)
   
    if not inplace:
        box = box.clone()
        
    box[:, [0, 2]] += 2 * (image_center[[0, 2]] - box[:, [0, 2]])
    box_w           = abs(box[:, 0] - box[:, 2])
    box[:, 0]      -= box_w
    box[:, 2]      += box_w
    return box


def horizontal_translate_box(
    box       : Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
    inplace   : bool        = False,
) -> Tensor:
    """Translate the bounding box in horizontal direction.

    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Original image size.
        magnitude (int): Horizontally translation.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Translated boxes of shape [N, 4].
    """
    return translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [magnitude, 0],
        center     = center,
        drop_ratio = drop_ratio,
        inplace    = inplace
    )


def nms(box: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Perform non-maxima suppression (NMS) on a given image of bounding boxes
    according to the intersection-over-union (IoU).
    
    NMS iteratively removes lower scoring boxes which have an IoU greater than
    `iou_threshold` with another (higher scoring) box.
    
    If multiple boxes have the exact same score and satisfy the IoU criterion
    with respect to a reference box, the selected box is not guaranteed to be
    the same between CPU and GPU. This is similar to the behavior of argsort in
    PyTorch when repeated values are present.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4] to perform NMS on.
            They are expected to be in (x1, y1, x2, y2) format with
            `0 <= x1 < x2` and `0 <= y1 < y2`.
        scores (Tensor): Scores for each one of the boxes.
        iou_threshold (float): Discards all overlapping boxes with
            iou > iou_threshold

    Return:
        Indices of the elements that have been kept by NMS, sorted in decreasing
        order of scores

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
    assert_tensor_of_ndim(box, 2)
    assert_tensor_of_ndim(scores, 1)
    
    if box.shape[-1] != 4:
        raise ValueError(
            f"`box` must have the shape of [N, 4]. But got: {box.shape}."
        )
    if box.shape[0] != scores.shape[0]:
        raise ValueError(
            f"`box` and `scores` must have same length. "
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


def rotate_box(
    box       : Tensor,
    image_size: Ints,
    angle     : float,
    center    : Ints | None = None,
    drop_ratio: float             = 0.0,
    inplace   : bool              = False,
) -> Tensor:
    """
    Rotate the bounding box by the given magnitude.

    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Original image size.
        angle (float): Angle to rotate the bounding box.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Translated boxes of shape [N, 4].
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
        inplace    = inplace,
    )


def scale_box(
    box       : Tensor,
    cur_size  : Ints,
    new_size  : Ints |   None = None,
    factor    : Floats | None = (1.0, 1.0),
    keep_shape: bool          = False,
    drop_ratio: float         = 0.0,
    inplace   : bool          = False,
) -> Tensor:
    """
    Scale bounding boxes coordinates by the given factor or by inferring from
    current image size and new size.
    
    References:
        https://blog.paperspace.com/data-augmentation-bounding-boxes-scaling-translation/

    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size (Ints): Current image size.
        new_size (Ints | None): New image size. Defaults to None.
        factor (Floats): Desired scaling factor in each direction. If scalar,
            the value is used for both the vertical and horizontal direction.
            Defaults to (1.0, 1.0).
        keep_shape (bool): When True, translate the scaled bounding boxes.
            Defaults to False.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Scaled bounding boxes of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    
    h0, w0 = to_size(cur_size)  # H, W
    
    if new_size is not None:
        h1, w1                 = to_size(new_size)  # H, W
        factor_ver, factor_hor = float(h1 / h0), float(w1 / w0)
    elif isinstance(factor, float):
        factor_ver = factor_hor = factor
        h1, w1     = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    else:
        factor_ver, factor_hor = factor
        h1, w1                 = int(h0 * factor_ver), int(w0 * factor_hor)  # H, W
    
    if not inplace:
        box = box.clone()
        
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
    box       : Tensor,
    cur_size  : Ints,
    new_size  : Ints,
    ratio_pad        = None,
    inplace   : bool = False,
) -> Tensor:
    """
    Scale bounding boxes coordinates (from detector size) to the original image
    size.

    Args:
        box (Tensor):Bounding boxes of shape [N, 4]. They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        cur_size (Ints): Detector's input size.
        new_size (Ints): Original image size.
        ratio_pad: Defaults to None.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Scaled bounding boxes of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    
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

    if not inplace:
        box = box.clone()

    box[:, [0, 2]] -= pad[0]  # x padding
    box[:, [1, 3]] -= pad[1]  # y padding
    box[:, :4]     /= gain
    return clip_box(box=box, image_size=new_size, inplace=inplace)


def shear_box(
    box       : Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
    inplace   : bool        = False,
) -> Tensor:
    """
    Shear bounding boxes coordinates by the given magnitude.
    
    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Original image size.
        magnitude (Ints): Shear magnitude.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Sheared bounding boxes of shape [N, 4].
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
        inplace    = inplace,
    )


def translate_box(
    box       : Tensor,
    image_size: Ints,
    magnitude : Ints,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
    inplace   : bool        = False,
) -> Tensor:
    """
    Translate the bounding box by the given magnitude.

    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Original image size.
        magnitude (Ints): Translation magnitude.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Translated boxes of shape [N, 4].
    """
    return affine_box(
        box        = box,
        image_size = image_size,
        angle      = 0.0,
        translate  = magnitude,
        scale      = 1.0,
        shear      = [0, 0],
        center     = center,
        drop_ratio = drop_ratio,
        inplace    = inplace,
    )
    

def vertical_flip_box(
    box: Tensor, image_center: Tensor, inplace: bool = False,
) -> Tensor:
    """
    Flip boxes vertically, which are specified by their (cx, cy, w, h) norm
    coordinates.
	
	Reference:
		https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
	
    Args:
        box (Tensor): Bounding boxes of shape [N, 4] to be flipped.
        image_center (Tensor): Center of the image.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Flipped boxes of shape [N, 4].
    """
    assert_tensor_of_ndim(box, 2)
    
    if not inplace:
        box = box.clone()
        
    box[:, [1, 3]] += 2 * (image_center[[0, 2]] - box[:, [1, 3]])
    box_h           = abs(box[:, 1] - box[:, 3])
    box[:, 1]      -= box_h
    box[:, 3]      += box_h
    return box


def vertical_translate_box(
    box       : Tensor,
    image_size: Ints,
    magnitude : int,
    center    : Ints | None = None,
    drop_ratio: float       = 0.0,
    inplace   : bool        = False
) -> Tensor:
    """
    Translate the bounding box in vertical direction.

    Args:
        box (Tensor): Bounding boxes of shape [N, 4]. They are expected to be
            in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        image_size (Ints): Original image size.
        magnitude (int): Vertically translation.
        center (Ints | None): Center of affine transformation. If None, use the
            center of the image. Defaults to None.
        drop_ratio (float): If the fraction of a bounding box left in the image
            after being clipped is less than `drop_ratio` the bounding box is
            dropped. If `drop_ratio==0`, don't drop any bounding boxes.
            Defaults to 0.0.
        inplace (bool): If True, make this operation inplace. Defaults to False.
            
    Returns:
        Translated boxes of shape [N, 4].
    """
    return translate_box(
        box        = box,
        image_size = image_size,
        magnitude  = [0, magnitude],
        center     = center,
        drop_ratio = drop_ratio,
        inplace    = inplace,
    )
