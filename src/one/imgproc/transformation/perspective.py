#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://mathworld.wolfram.com/topics/GeometricTransformations.html

List of operation:
    - Cantellation
    - Central Dilation
    - Collineation
    - Dilation
    - Elation
    - Elliptic Rotation
    - Expansion
    - Geometric Correlation
    - Geometric Homology
    - Harmonic Homology
    - Homography
    - Perspective Collineation
    - Polarity
    - Projective Collineation
    - Projective Correlation
    - Projectivity
    - Stretch
    - Twirl
    - Unimodular Transformation
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import cv2
import numpy as np

from one.core import Int2T
from one.imgproc.spatial import is_box_candidates

__all__ = [
    "image_box_random_perspective",
    "paired_images_random_perspective",
    "random_perspective",
]


# MARK: - Functional

def paired_images_random_perspective(
    image1     : np.ndarray,
    image2     : np.ndarray = (),
    rotate     : float      = 10,
    translate  : float      = 0.1,
    scale      : float      = 0.1,
    shear      : float      = 10,
    perspective: float      = 0.0,
    border     : Sequence   = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask.

    Args:
        image1 (np.ndarray):
            Image.
        image2 (np.ndarray):
            Mask.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image1_new (np.ndarray):
            Augmented image.
        image2_new (np.ndarray):
            Augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height     = image1.shape[0] + border[0] * 2  # Shape of [HWC]
    width      = image1.shape[1] + border[1] * 2
    image1_new = image1.copy()
    image2_new = image2.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image1_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image1_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image1_new = cv2.warpPerspective(
                image1_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
            image2_new  = cv2.warpPerspective(
                image2_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image1_new = cv2.warpAffine(
                image1_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
            image2_new  = cv2.warpAffine(
                image2_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
    
    return image1_new, image2_new


def image_box_random_perspective(
    image      : np.ndarray,
    box        : np.ndarray = (),
    rotate     : float      = 10,
    translate  : float      = 0.1,
    scale      : float      = 0.1,
    shear      : float      = 10,
    perspective: float      = 0.0,
    border     : Int2T      = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    r"""Perform random perspective the image and the corresponding bounding box
    labels.

    Args:
        image (np.ndarray):
            Image of shape [H, W, C].
        box (np.ndarray):
            Bounding box labels where the box coordinates are located at:
            labels[:, 2:6]. Default: `()`.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (sequence):

    Returns:
        image_new (np.ndarray):
            Augmented image.
        box_new (np.ndarray):
            Augmented bounding boxes.
    """
    height    = image.shape[0] + border[0] * 2  # Shape of [H, W, C]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    box_new   = box.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T       = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image_new = cv2.warpPerspective(
                image_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image_new = cv2.warpAffine(
                image_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )

    # NOTE: Transform bboxes' coordinates
    n = len(box_new)
    if n:
        # NOTE: Warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = box_new[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)
        # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # Transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # Rescale
        else:  # Affine
            xy = xy[:, :2].reshape(n, 8)
        
        # NOTE: Create new boxes
        x  = xy[:, [0, 2, 4, 6]]
        y  = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T
        
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        
        # NOTE: Filter candidates
        i = is_box_candidates(box_new[:, 2:6].T * s, xy.T)
        box_new = box_new[i]
        box_new[:, 2:6] = xy[i]
    
    return image_new, box_new


def random_perspective(
    image      : np.ndarray,
    rotate     : float    = 10,
    translate  : float    = 0.1,
    scale      : float    = 0.1,
    shear      : float    = 10,
    perspective: float    = 0.0,
    border     : Sequence = (0, 0)
) -> tuple[np.ndarray, np.ndarray]:
    """Perform random perspective the image and the corresponding mask labels.

    Args:
        image (np.ndarray):
            Image.
        rotate (float):
            Image rotation (+/- deg).
        translate (float):
            Image translation (+/- fraction).
        scale (float):
            Image scale (+/- gain).
        shear (float):
            Image shear (+/- deg).
        perspective (float):
            Image perspective (+/- fraction), range 0-0.001.
        border (tuple, list):

    Returns:
        image_new (np.ndarray):
            Augmented image.
        mask_labels_new (np.ndarray):
            Augmented mask.
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    
    height    = image.shape[0] + border[0] * 2  # Shape of [HWC]
    width     = image.shape[1] + border[1] * 2
    image_new = image.copy()
    
    # NOTE: Center
    C       = np.eye(3)
    C[0, 2] = -image_new.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image_new.shape[0] / 2  # y translation (pixels)
    
    # NOTE: Perspective
    P       = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    # NOTE: Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-rotate, rotate)
    # Add 90deg rotations to small rotations
    # a += random.choice([-180, -90, 0, 90])
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # NOTE: Shear
    S       = np.eye(3)
    # x shear (deg)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # NOTE: Translation
    T       = np.eye(3)
    # x translation (pixels)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    # y translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # NOTE: Combined rotation matrix
    M = T @ S @ R @ P @ C  # Order of operations (right to left) is IMPORTANT
    # Image changed
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image_new = cv2.warpPerspective(
                image_new, M, dsize=(width, height),
                borderValue=(114, 114, 114)
            )
        else:  # Affine
            image_new = cv2.warpAffine(
                image_new, M[:2], dsize=(width, height),
                borderValue=(114, 114, 114)
            )
    
    return image_new
