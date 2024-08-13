#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the pipeline to generate synthetic data."""

from __future__ import annotations

import random
from typing import Any

import click
import cv2
import numpy as np

import mon
import math

random.seed(0)


# region Function

def random_patch_image_box(
    canvas       : np.ndarray,
    patches      : Any,
    masks        : Any                 = None,
    ids          : list[int]           = None,
    scale        : float | list[float] = [1.0, 1.0],
    gamma        : float | list[float] = 0.0,
    iou_threshold: float               = 0.0,
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
        iou_threshold: An IOU threshold.
        
    Returns:
        A generated image.
        Bounding boxes.
    """
    if not type(patches) == type(masks):
        raise TypeError(
            f"patches and masks must have the same type, but got "
            f"{type(patches)} and {type(masks)}."
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
    
    if masksImageDataset:
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
    
    # canvas = mon.adjust_gamma(image=canvas, gamma=0.5)
    h, w   = mon.get_image_size(input=canvas)
    box    = np.zeros(shape=[len(patches), 5], dtype=np.float64)
    for i, p in enumerate(patches):
        # Random scale
        s          = random.uniform(*scale)
        p_h0, p_w0 = mon.vision.core.image.get_image_size(p)
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
            
            if idsImageDataset:
                b = np.array([ids[i], x1, y1, x2, y2], dtype=np.float64)
            else:
                b = np.array([-1, x1, y1, x2, y2], dtype=np.float64)
                
            max_iou = max([mon.bbox_iou(b[1:5], j[1:5]) for j in box])
            if max_iou <= iou_threshold:
                box[i] = b
                break
            
            tries += 1
            # if tries == 10:
                # iou_threshold += 0.1
        
        # Blend patches into canvas
        p_blur     = cv2.medianBlur(p, 3)  # Blur to remove noise around the edges of objects
        p_gray     = cv2.cvtColor(p_blur, cv2.COLOR_RGB2GRAY)
        ret, masks = cv2.threshold(p_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv   = cv2.bitwise_not(masks)
        bg         = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg         = cv2.bitwise_and(p,   p,   mask=masks)
        dst        = cv2.add(bg, fg)
        roi[:]     = dst
    
    # Adjust brightness via Gamma correction
    canvas = canvas.astype(dtype=np.uint8)
    # canvas = mon.adjust_gamma(image=canvas, gamma=random.uniform(gamma[0], gamma[1]))
    # canvas = canvas.astype(dtype=np.uint8)
    return canvas, box


@click.command()
@click.option("--image-dir",         default=mon.DATA_DIR/"aic23-autocheckout/train/patches-denoise-deblur", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",         default=mon.DATA_DIR/"aic23-autocheckout/train/segmentation-labels", type=click.Path(exists=True), help="Binary mask directory.")
@click.option("--background-dir",    default=mon.DATA_DIR/"aic23-autocheckout/train/backgrounds", type=click.Path(exists=True), help="Background image directory.")
@click.option("--output-dir",        default=mon.DATA_DIR/"aic23-autocheckout/train/synthetic-03", type=click.Path(exists=False), help="Output directory.")
@click.option("--patches-per-image", default=3,     type=int, help="Number of patches to place on an image.")
@click.option("--iou-threshold",     default=0.0,   type=float)
@click.option("--bbox-format",       default="voc", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--verbose",           is_flag=True)
def gen_synthetic_image(
    image_dir        : mon.Path,
    label_dir        : mon.Path,
    background_dir   : mon.Path,
    output_dir       : mon.Path,
    patches_per_image: int,
    iou_threshold    : float,
    bbox_format      : str,
    verbose          : bool
):
    assert image_dirImageDataset and mon.Path(image_dir).is_dir()
    assert background_dirImageDataset and mon.Path(background_dir).is_dir()
    assert label_dirImageDataset and mon.Path(label_dir).is_dir()
    
    image_dir           = mon.Path(image_dir)
    background_dir      = mon.Path(background_dir)
    synthetic_image_dir = output_dir/"images" or image_dir.parent/"synthetic/images"
    synthetic_image_dir = mon.Path(synthetic_image_dir)
    synthetic_bbox_dir  = synthetic_image_dir.parent / f"labels-{bbox_format}"
    synthetic_bbox_dir  = mon.Path(synthetic_bbox_dir)
    mon.mkdirs(paths=[synthetic_image_dir, synthetic_bbox_dir], parents=True, exist_ok=True)
    
    image_files = []
    masks_files = []
    sizes       = {}
    with mon.get_progress_bar() as pbar:
        for f in pbar.track(
            sequence    = list(image_dir.glob("*")),
            description = f"[bright_yellow] Listing files"
        ):
            file_name = str(f).split(".")[0]
            seg_file  = f"{file_name}_seg.jpg"
            seg_file  = seg_file.replace(str(f.parent.name), "segmentation-labels")
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
        patches_per_image = int(patches_per_image)
        total    = math.floor(len(image_files) / patches_per_image)
        item_idx = 0
        for i in pbar.track(
            sequence    = range(total),
            description = f"[bright_yellow]Generating data"
        ):
            images     = []
            masks      = []
            ids        = []
            bg_idx     = random.randint(0, len(background_files))
            background = mon.read_image(background_files[0], to_rgb=True)
            
            for j in range(item_idx, item_idx + patches_per_image):
                id    = int(str(image_files[j].stem).split("_")[0]) - 1
                image = mon.read_image(image_files[j], to_rgb=True)
                mask  = mon.read_image(masks_files[j], to_rgb=True)
                ids.append(id)
                images.append(image)
                masks.append(mask)
                if verbose:
                    mon.console.log(image_files[j])
                
            synthetic, boxes = random_patch_image_box(
                canvas        = background,
                patches       = images,
                masks         = masks,
                ids           = ids,
                scale         = [1.0, 1.0],
                gamma         = [0.9, 1.0],
                iou_threshold = iou_threshold,
            )
            synthetic      = cv2.cvtColor(synthetic, cv2.COLOR_BGR2RGB)
            synthetic_file = str(synthetic_image_dir / f"{i:06}.jpg")
            cv2.imwrite(synthetic_file, synthetic)
            
            h, w = mon.get_image_size(input=synthetic)
            if bbox_format in ["coco"]:
                boxes[:, 1:5] = mon.bbox_xyxy_to_xywh(boxes[:, 1:5])
            elif bbox_format in ["yolo"]:
                boxes[:, 1:5] = mon.bbox_xyxy_to_cxcywhn(boxes[:, 1:5], h, w)
            
            synthetic_bbox_file = str(synthetic_bbox_dir / f"{i:06}.txt")
            with open(synthetic_bbox_file, "w") as f:
                for b in boxes:
                    f.write(f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}\n")
            
            item_idx += patches_per_image
    
# endregion


# region Main

if __name__ == "__main__":
    gen_synthetic_image()

# endregion
