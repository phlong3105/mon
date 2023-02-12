#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the pipeline to generate synthetic data."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

import cv2
import numpy as np
from joblib import delayed, Parallel

import mon
from mon.foundation import math

random.seed(0)


# region Function

def random_patch_image_box(
    canvas   : np.ndarray,
    patches  : Any,
    masks    : Any                 = None,
    ids      : list[int]           = None,
    scale    : float | list[float] = [1.0, 1.0],
    gamma    : float | list[float] = 0.0,
    iou_thres: float               = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly place patches of small images over a large background image.
    Also, generate bounding bbox for each patch and add some basic augmentation.
    
    References:
        `<https://datahacker.rs/012-blending-and-pasting-images-using-opencv/>`_
    
    Args:
        canvas: A background image.
        patches: A collection of small image patches.
        masks: A collection of binary masks accompanying :param:`patch`.
        ids: A list of bounding boxes' IDs.
        gamma: Gamma correction parameter [1.0-3.0] to adjust the brightness in
            the image.
        
        angle: A rotation angle.
        scale: A scale factor.
        iou_thres: An IOU threshold.
        
    Returns:
        A generated image.
        Bounding boxes.
    """
    if not type(patches) == type(masks):
        raise TypeError(
            f"patches and masks must have the same type, but got "
            f"{type(patches)} and {type(masks)}."
        )
    if not patches.shape == masks.shape:
        raise TypeError(
            f"patches and masks must have the same shape, but got "
            f"{patches.shape} and {masks.shape}."
        )
    
    patches = list(patches)
    masks   = list(masks)
    if not len(patches) == len(masks):
        raise TypeError(
            f"patches and masks must have the same length, but got "
            f"{len(patches)} and {len(masks)}."
        )
    
    if isinstance(scale, int | float):
        scale = [scale, scale]
    if len(scale) == 1:
        scale = [scale, scale]
    assert isinstance(scale, list)
    
    if isinstance(gamma, int | float):
        gamma = [-gamma, gamma]
    if len(gamma) == 1:
        gamma = [-gamma[0], gamma[0]]
    assert isinstance(gamma, list)
    
    if masks is not None:
        # for i, (p, m) in enumerate(zip(patch, mask)):
        #     cv2.imwrite(f"{i}_image.png", p[:, :, ::-1])
        #     cv2.imwrite(f"{i}_mask.png",  m[:, :, ::-1])
        patches = [cv2.bitwise_and(p, m) for p, m in zip(patches, masks)]
        # for i, p in enumerate(patch):
        #     cv2.imwrite(f"{i}_patch.png", p[:, :, ::-1])
    
    if isinstance(ids, list):
        if not len(ids) == len(patches):
            raise TypeError(
                f"ids and patches must have the same length, but got "
                f"{len(ids)} and {len(patches)}."
            )

    canvas = mon.adjust_gamma(image=canvas, gamma=2.0)
    h, w   = mon.get_image_size(image=canvas)
    box    = np.zeros(shape=[len(patches), 5], dtype=np.float)
    for i, p in enumerate(patches):
        # Random scale
        s          = random.uniform(*scale)
        p_h0, p_w0 = mon.get_image_size(p)
        p_h1, p_w1 = (int(p_h0 * s), int(p_w0 * s))
        p          = cv2.resize(p, (p_h1, p_w1))
        # Random rotate
        # cx, cy     = (p_h1 // 2, p_w1 // 2)
        # m          = cv2.getRotationMatrix2D((cx, cy), random.randint(*angle), 1.0)
        # p          = cv2.warpAffine(image, m, (p_w1, p_h1))
        # p          = mon.crop_zero_region(p)
        
        p_h, p_w  = mon.get_image_size(p)
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

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",             type=str, default=mon.DATA_DIR, help="Image directory.")
    parser.add_argument("--background",        type=str, default=mon.DATA_DIR, help="Background image directory.")
    parser.add_argument("--patches-per-image", type=int, default=7, help="Number of patches to place on an image.")
    parser.add_argument("--output",            type=str, default=mon.DATA_DIR, help="Output directory.")
    parser.add_argument("--bbox-format",       type=str, default="yolo", help="Bounding bbox format: coco (xywh), voc (xyxy), yolo (cxcywhn).")
    parser.add_argument("--verbose",           action="store_true")
    args = parser.parse_args()
    return args

# endregion


# region Main

if __name__ == "__main__":
    args = parse_args()

    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.background is not None and mon.Path(args.background).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    
    bbox_format        = args.bbox_format
    image_dir          = mon.Path(args.image)
    background_dir     = mon.Path(args.background)
    synthetic_dir      = args.output or image_dir.parent / "synthetic"
    synthetic_dir      = mon.Path(synthetic_dir)
    synthetic_bbox_dir = args.output or image_dir.parent / f"synthetic-bboxes-{bbox_format}"
    synthetic_bbox_dir = mon.Path(synthetic_bbox_dir)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    masks_files = []
    sizes       = {}
    with mon.get_progress_bar() as pbar:
        for f in pbar.track(
            sequence    = list(image_dir.rglob("*")),
            description = f"[bright_yellow] Visualizing"
        ):
            file_name = str(f).split(".")[0]
            seg_file  = f"{file_name}_seg.jpg"
            seg_file  = seg_file.replace("patches", "segmentation_labels")
            seg_file  = mon.Path(seg_file)
            if f.is_image_file() and seg_file.is_image_file():
                image_files.append(f)
                masks_files.append(seg_file)
    
    # Shuffle
    shuffle          = mon.shuffle_dict(dict(zip(image_files, masks_files)))
    image_files      = list(shuffle.keys())
    masks_files      = list(shuffle.values())
    background_files = list(background_dir.rglob("*"))
    
    # Generate train images and labels for training YOLO models
    with mon.get_progress_bar() as pbar:
        patches_per_image = int(args.patches_per_image)
        item_idx = 0
        total    = math.floor(len(image_files) / patches_per_image)
        task     = pbar.add_task(f"[bright_yellow]Generating data", total=total)
        
        def patch_image(i):
            global item_idx
            images     = []
            masks      = []
            ids        = []
            bg_idx     = random.randint(0, len(background_files))
            background = mon.read_image(background_files[bg_idx], to_rgb=True)
            
            for j in range(item_idx, item_idx + patches_per_image):
                id    = int(str(image_files[j].stem).split("_")[0]) - 1
                image = mon.read_image(image_files[j], to_rgb=True)
                mask  = mon.read_image(masks_files[j], to_rgb=True)
                ids.append(id)
                images.append(image)
                masks.append(mask)
                
            synthetic, boxes = random_patch_image_box(
                canvas    = background,
                patches   = images,
                masks     = masks,
                ids       = ids,
                scale     = [1.0, 1.0],
                gamma     = [0.9, 1.0],
                iou_thres = 0.10,
            )
            synthetic      = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
            synthetic_file = synthetic_dir / f"{i:06}.jpg"
            cv2.imwrite(synthetic_file, synthetic)
            
            h, w = mon.get_image_size(image=synthetic)
            if bbox_format in ["coco"]:
                boxes[:, 1:5] = mon.bbox_xyxy_to_xywh(boxes[:, 1:5])
            elif bbox_format in ["yolo"]:
                boxes[:, 1:5] = mon.bbox_xyxy_to_cxcywhn(boxes[:, 1:5], h, w)
            
            synthetic_bbox_file = synthetic_bbox_dir / f"{i:06}.txt"
            with open(synthetic_bbox_file, "w") as f:
                for b in boxes:
                    f.write(f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}\n")
            
            item_idx += patches_per_image
            pbar.update(task, advance=1)
            
        Parallel(n_jobs=os.cpu_count(), require="sharedmem")(
            delayed(patch_image)(i) for i in range(total)
        )
