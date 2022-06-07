#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Randomly place patches of small images over a large background image.
"""

from __future__ import annotations

from copy import copy
from random import randint
from random import uniform
from typing import Optional

import cv2
import numpy as np
from torch import Tensor

from one.core import FloatAnyT
from one.core import get_image_size
from one.core import ListOrTupleAnyT
from one.core import ScalarOrCollectionAnyT
from one.core import TensorOrArray
from one.imgproc.filtering import adjust_gamma
from one.imgproc.spatial import compute_single_box_iou
from one.imgproc.transformation.crop import crop_zero_region
from one.imgproc.transformation.resize import resize
from one.imgproc.transformation.rotation import rotate

__all__ = [
    "random_patch_numpy_image_box",
]


# MARK: - Functional

# TODO: Right now we just provide support for numpy.
def random_patch_numpy_image_box(
    canvas : TensorOrArray,
    patch  : ScalarOrCollectionAnyT[TensorOrArray],
    mask   : Optional[ScalarOrCollectionAnyT[TensorOrArray]] = None,
    id     : Optional[ListOrTupleAnyT[int]]                  = None,
    angle  : FloatAnyT = (0, 0),
    scale  : FloatAnyT = (1.0, 1.0),
    gamma  : FloatAnyT = (1.0, 1.0),
    overlap: float     = 0.1,
) -> tuple[TensorOrArray, TensorOrArray]:
    """Randomly place patches of small images over a large background image and
    generate accompany bounding boxes. Also, add some basic augmentation ops.
    
    References:
        https://datahacker.rs/012-blending-and-pasting-images-using-opencv/
    
    Args:
        canvas (TensorOrArray[C, H, W]):
            Background image to place patches over.
        patch (ScalarOrCollectionAnyT[TensorOrArray]):
            Collection of TensorOrArray[C, H, W] or a TensorOrArray[B, C, H, W]
            of small images.
        mask (ScalarOrCollectionAnyT[TensorOrArray]):
            Collection of TensorOrArray[C, H, W] or a TensorOrArray[B, C, H, W]
            of interested objects' masks in small images (black and white image).
        id (ListOrTupleAnyT[int], optional):
            Bounding boxes' IDs.
        angle (FloatAnyT):
            Patches will be randomly rotated with angle in degree between
            `angle[0]` and `angle[1]`, clockwise direction. Default: `(0.0, 0.0)`.
        scale (FloatAnyT):
            Patches will be randomly scaled with factor between `scale[0]` and
            `scale[1]`. Default: `(1.0, 1.0)`.
        gamma (FloatAnyT):
            Gamma correction value used to augment the brightness of the objects
            between `gamma[0]` and `gamma[1]`. Default: `(1.0, 1.0)`.
        overlap (float):
            Overlapping ratio threshold.
            
    Returns:
        gen_image (TensorOrArray[C, H, W]):
            Generated image.
        box (TensorOrArray[N, 5], optional):
            Bounding boxes of small patches. Boxes are expected to be in
            (id, x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """
    if type(patch) != type(mask):
        raise TypeError(f"`patch` and `mask` must have the same type. "
                        f"But got: {type(patch)} != {type(mask)}.")
    if isinstance(patch, (Tensor, np.ndarray)) and isinstance(mask, (Tensor, np.ndarray)):
        if patch.shape != mask.shape:
            raise ValueError(f"`patch` and `mask` must have the same shape. "
                             f"But got: {patch.shape} != {mask.shape}.")
        patch = list(patch)
        mask  = list(mask)
    if isinstance(patch, (list, tuple)) and isinstance(mask, (list, tuple)):
        if len(patch) != len(mask):
            raise ValueError(f"`patch` and `mask` must have the same length. "
                             f"But got: {len(patch)} != {len(mask)}.")
    
    if isinstance(angle, (int, float)):
        angle = [-int(angle), int(angle)]
    if len(angle) == 1:
        angle = [-angle[0], angle[0]]
    
    if isinstance(scale, (int, float)):
        scale = [float(scale), float(scale)]
    if len(scale) == 1:
        scale = [scale, scale]

    if isinstance(gamma, (int, float)):
        gamma = [0.0, float(angle)]
    if len(gamma) == 1:
        gamma = [0.0, gamma]
    if not 0 < gamma[1] <= 1.0:
        raise ValueError(f"`gamma` must be between 0.0 and 1.0.")
    
    if mask is not None:
        # for i, (p, m) in enumerate(zip(patch, mask)):
        #     cv2.imwrite(f"{i}_image.png", p[:, :, ::-1])
        #     cv2.imwrite(f"{i}_mask.png",  m[:, :, ::-1])
        patch = [cv2.bitwise_and(p, m) for p, m in zip(patch, mask)]
        # for i, p in enumerate(patch):
        #     cv2.imwrite(f"{i}_patch.png", p[:, :, ::-1])

    if isinstance(id, (list, tuple)):
        if len(id) != len(patch):
            raise ValueError(f"`id` and `patch` must have the same length. "
                             f"But got: {len(id)} != {len(patch)}.")
    
    canvas = copy(canvas)
    canvas = adjust_gamma(canvas, 2.0)
    h, w   = get_image_size(canvas)
    box    = np.zeros(shape=[len(patch), 5], dtype=np.float)
    for i, p in enumerate(patch):
        # Random scale
        s          = uniform(scale[0], scale[1])
        p_h0, p_w0 = get_image_size(p)
        p_h1, p_w1 = (int(p_h0 * s), int(p_w0 * s))
        p          = resize(image=p, size=(p_h1, p_w1))
        # Random rotate
        p          = rotate(p, angle=randint(angle[0], angle[1]), keep_shape=False)
        # p          = ndimage.rotate(p, randint(angle[0], angle[1]))
        p          = crop_zero_region(p)
        p_h, p_w   = get_image_size(p)
        # cv2.imwrite(f"{i}_rotate.png", p[:, :, ::-1])
        
        # Random place patch in canvas. Set ROI's x, y position.
        tries     = 0
        iou_thres = overlap
        while tries <= 10:
            x1  = randint(0, w - p_w)
            y1  = randint(0, h - p_h)
            x2  = x1 + p_w
            y2  = y1 + p_h
            roi = canvas[y1:y2, x1:x2]
            
            if id is not None:
                b = np.array([id[i], x1, y1, x2, y2], dtype=np.float)
            else:
                b = np.array([-1, x1, y1, x2, y2], dtype=np.float)
                
            max_iou = max([compute_single_box_iou(b[1:5], j[1:5]) for j in box])
            if max_iou <= iou_thres:
                box[i] = b
                break
            
            tries += 1
            if tries == 10:
                iou_thres += 0.1
        
        # Blend patch into canvas
        p_blur    = cv2.medianBlur(p, 3)  # Blur to remove noise around the edges of objects
        p_gray    = cv2.cvtColor(p_blur, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(p_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv  = cv2.bitwise_not(mask)
        bg        = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg        = cv2.bitwise_and(p,   p,   mask=mask)
        dst       = cv2.add(bg, fg)
        roi[:]    = dst
        # cv2.imwrite(f"{i}_gray.png", p_gray)
        # cv2.imwrite(f"{i}_threshold.png", mask)
        # cv2.imwrite(f"{i}_maskinv.png", mask_inv)
        # cv2.imwrite(f"{i}_bg.png", bg[:, :, ::-1])
        # cv2.imwrite(f"{i}_fg.png", fg[:, :, ::-1])
        # cv2.imwrite(f"{i}_dst.png", dst[:, :, ::-1])
    
    # Adjust brightness via Gamma correction
    g      = uniform(gamma[0], gamma[1])
    canvas = adjust_gamma(canvas, g)
    return canvas, box
