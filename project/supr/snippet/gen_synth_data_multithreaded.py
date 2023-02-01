#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the pipeline to generate synthetic data."""

from __future__ import annotations

import copy
import glob
import math
import os.path
import random
from typing import Any

import cv2
import numpy as np
import torch
from joblib import delayed, Parallel

import mon
from mon.typing import Floats, Ints

random.seed(0)


# region Function

def random_patch_image_box(
    canvas   : torch.Tensor | np.ndarray,
    patches  : Any,
    masks    : Any    = None,
    ids      : Ints   = None,
    angle    : Floats = (0, 0),
    scale    : Floats = (1.0, 1.0),
    gamma    : Floats = (1.0, 1.0),
    iou_thres: float  = 0.1,
) -> tuple[
    torch.Tensor | np.ndarray,
    torch.Tensor | np.ndarray
]:
    """Randomly place patches of small images over a large background image.
    Also, generate bounding bbox for each patch and add some basic augmentation.
    
    References:
        https://datahacker.rs/012-blending-and-pasting-images-using-opencv/
    
    Args:
        canvas: A background image.
        patches: A collection of small image patches.
        masks: A collection of binary masks accompanying :param:`patch`.
        ids: A list of bounding boxes' IDs.
        angle: A rotation angle.
        scale: A scale factor.
        gamma: A gamma correction value used to augment the brightness.
        iou_thres: An IOU threshold.
        
    Returns:
        A generated image.
        Bounding boxes.
    """
    assert type(patches) == type(masks)
    assert patches.shape == masks.shape
    patches = list(patches)
    masks   = list(masks)
    assert len(patches) == len(masks)
    
    if isinstance(angle, (int, float)):
        angle = [-int(angle), int(angle)]
    if len(angle) == 1:
        angle = [-angle[0], angle[0]]
    
    if isinstance(scale, (int, float)):
        scale = [float(scale), float(scale)]
    if len(scale) == 1:
        scale = [scale, scale]
    assert isinstance(scale, list | tuple)
    
    if isinstance(gamma, (int, float)):
        gamma = [0.0, float(gamma)]
    if len(gamma) == 1:
        gamma = [0.0, gamma]
    assert isinstance(gamma, list | tuple)
    assert 0 < gamma[1] <= 1.0
    
    if masks is not None:
        # for i, (p, m) in enumerate(zip(patch, mask)):
        #     cv2.imwrite(f"{i}_image.png", p[:, :, ::-1])
        #     cv2.imwrite(f"{i}_mask.png",  m[:, :, ::-1])
        patches = [cv2.bitwise_and(p, m) for p, m in zip(patches, masks)]
        # for i, p in enumerate(patch):
        #     cv2.imwrite(f"{i}_patch.png", p[:, :, ::-1])
    
    if isinstance(ids, (list, tuple)):
        assert len(ids) == len(patches)
    
    canvas = copy.copy(canvas)
    canvas = mon.adjust_gamma(image=canvas, gamma=2.0)
    h, w   = mon.get_image_size(canvas)
    box    = np.zeros(shape=[len(patches), 5], dtype=np.float)
    for i, p in enumerate(patches):
        # Random scale
        s          = random.uniform(scale[0], scale[1])
        p_h0, p_w0 = mon.get_image_size(p)
        p_h1, p_w1 = (int(p_h0 * s), int(p_w0 * s))
        p          = mon.resize(image=p, size=(p_h1, p_w1))
        # Random rotate
        p          = mon.rotate(p, angle=random.randint(angle[0], angle[1]), keep_shape=False)
        # p          = ndimage.rotate(p, randint(angle[0], angle[1]))
        p          = mon.crop_zero_region(p)
        p_h, p_w   = mon.get_image_size(p)
        # cv2.imwrite(f"{i}_rotate.png", p[:, :, ::-1])
        
        # Random place patch in canvas. Set ROI's x, y position.
        tries = 0
        while tries <= 10:
            x1  = random.randint(0, w - p_w)
            y1  = random.randint(0, h - p_h)
            x2  = x1 + p_w
            y2  = y1 + p_h
            roi = canvas[y1:y2, x1:x2]
            
            if ids is not None:
                b = np.array([ids[i], x1, y1, x2, y2], dtype=np.float)
            else:
                b = np.array([-1, x1, y1, x2, y2], dtype=np.float)
                
            max_iou = max([mon.get_single_bbox_iou(b[1:5], j[1:5]) for j in box])
            if max_iou <= iou_thres:
                box[i] = b
                break
            
            tries += 1
            if tries == 10:
                iou_thres += 0.1
        
        # Blend patches into canvas
        p_blur    = cv2.medianBlur(p, 3)  # Blur to remove noise around the edges of objects
        p_gray    = cv2.cvtColor(p_blur, cv2.COLOR_RGB2GRAY)
        ret, masks = cv2.threshold(p_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv  = cv2.bitwise_not(masks)
        bg        = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg        = cv2.bitwise_and(p,   p,   mask=masks)
        dst       = cv2.add(bg, fg)
        roi[:]    = dst
        # cv2.imwrite(f"{i}_gray.png", p_gray)
        # cv2.imwrite(f"{i}_threshold.png", mask)
        # cv2.imwrite(f"{i}_maskinv.png", mask_inv)
        # cv2.imwrite(f"{i}_bg.png", bg[:, :, ::-1])
        # cv2.imwrite(f"{i}_fg.png", fg[:, :, ::-1])
        # cv2.imwrite(f"{i}_dst.png", dst[:, :, ::-1])
    
    # Adjust brightness via Gamma correction
    canvas = mon.adjust_gamma(image=canvas, gamma=random.uniform(gamma[0], gamma[1]))
    return canvas, box


def draw_rect(drawing: np.ndarray, box: np.ndarray, color: Ints | Any = None):
    drawing = drawing.copy()
    box     = box[:, :4]
    box     = box.reshape(-1, 4)
    color   = color or [255, 255, 255]

    for cord in box:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
        pt1      = int(pt1[0]), int(pt1[1])
        pt2      = int(pt2[0]), int(pt2[1])
        drawing = cv2.rectangle(drawing.copy(), pt1, pt2, color, 3)
    return drawing

# endregion


# region Main

image_file_pattern = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "images", "*")
generate_image_dir = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "generate_images")
generate_label_dir = os.path.join(datasets_dir, "aicity", "aic22retail", "train", "generate_labels")
create_dirs([generate_image_dir, generate_label_dir])

# NOTE: Get list of image sizes
image_files        = []
segmentation_files = []
sizes              = {}

with progress_bar() as pbar:
    for image_file in pbar.track(
        glob.glob(image_file_pattern),
        description=f"[bright_yellow]Listing files"
    ):
        file_name         = image_file.split(".")[0]
        segmentation_file = f"{file_name}_seg.jpg"
        segmentation_file = segmentation_file.replace("images", "segmentation_labels")
        
        if is_image_file(image_file) and is_image_file(segmentation_file):
            # image        = read_image(image_file,        VisionBackend.CV)
            # segmentation = read_image(segmentation_file, VisionBackend.CV)
            # cv2.imshow("image", image)
            # cv2.imshow("segmentation", segmentation)
            # cv2.waitKey(0)
            image_files.append(image_file)
            segmentation_files.append(segmentation_file)
            # sizes[image_file] = get_image_size(image)

# NOTE: Shuffle
d = {}
for image, segment in zip(image_files, segmentation_files):
    d[image] = segment

keys = list(d.keys())
random.shuffle(keys)
shuffled_d = {}
for key in keys:
    shuffled_d[key] = d[key]

image_files        = list(shuffled_d.keys())
segmentation_files = list(shuffled_d.values())

# NOTE: Generate train images and labels for training YOLO models.
background          = read_image("../data/background.png", VisionBackend.CV)
num_items           = 7
num_generate_images = math.floor(len(image_files) / num_items)

with progress_bar() as pbar:
    item_idx = 0
    total    = num_generate_images
    task     = pbar.add_task(f"[bright_yellow]Generating data", total=total)
    
    def patch_image(i):
        global item_idx
        images   = []
        segments = []
        ids      = []
        for j in range(item_idx, item_idx + num_items):
            ids.append(int(os.path.basename(image_files[j]).split("_")[0]) - 1)
            images.append(read_image(image_files[j], VisionBackend.CV))
            segments.append(read_image(segmentation_files[j], VisionBackend.CV))
        
        gen_image, boxes = random_patch_numpy_image_box(
            canvas  = background.copy(),
            patch   = images,
            mask    = segments,
            id      = ids,
            angle   = [0, 360],
            scale   = [1.0, 1.0],
            gamma   = [0.9, 1.0],
            overlap = 0.10,
        )
        gen_image      = gen_image[:, :, ::-1]
        gen_image_file = os.path.join(generate_image_dir, f"{i:06}.jpg")
        cv2.imwrite(gen_image_file, gen_image)
        
        h, w           = get_image_hw(gen_image)
        boxes[:, 1:5]  = box_xyxy_to_cxcywh_norm(boxes[:, 1:5], h, w)
        gen_label_file = os.path.join(generate_label_dir, f"{i:06}.txt")
    
        with open(gen_label_file, "w") as f:
            for b in boxes:
                f.write(f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}\n")
        
        item_idx += num_items
        pbar.update(task, advance=1)
        
    Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
        delayed(patch_image)(i) for i in range(total)
    )
