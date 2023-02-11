#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the visualization of bounding boxes on images."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import mon


# region Function

def visualize_bboxes(args: dict):
    """Visualize bounding boxes on images."""
    assert args["image"] is not None and mon.Path(args["image"]).is_dir()
    assert args["label"] is not None and mon.Path(args["label"]).is_dir()
    
    from_format = args["bbox_format"]
    save_image  = args["save_image"]
    verbose     = args["verbose"]
    
    image_dir   = mon.Path(args["image"])
    label_dir   = mon.Path(args["label"])
    output_dir  = args["output"] or label_dir.parent / "visualize"
    output_dir  = mon.Path(output_dir)
    if save_image:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Visualizing"
        ):
            image      = cv2.imread(str(image_files[i]))
            h, w, c    = image.shape
            label_file = label_dir / f"{image_files[i].stem}.txt"
            with open(label_file, "r") as in_file:
                l = in_file.read().splitlines()
                l = [x.strip().split(" ") for x in l]
                l = [x for x in l if len(x) >= 5]
                b = np.array([list(map(float, x[1:])) for x in l])
                
                if from_format in ["coco"]:
                    b = mon.bbox_xywh_to_xyxy(bbox=b)
                elif from_format in ["yolo"]:
                    b = mon.bbox_cxcywhn_to_xyxy(bbox=b, height=h, width=w)
                
                colors = mon.RGB.values()
                n      = len(colors)
                for j, x in enumerate(b):
                    mon.draw_bbox(
                        image     = image,
                        bbox      = x,
                        color     = colors[abs(hash(l[j][0])) % n],
                        thickness = 2,
                    )
                
                image = cv2.putText(
                    img       = image,
                    text      = f"{image_files[i].stem}",
                    org       = [50, 50],
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color     = [255, 255, 255],
                    thickness = 2,
                    lineType  = cv2.LINE_AA,
                )
                if save_image:
                    output_file = output_dir / f"{image_files[i].name}"
                    cv2.imwrite(str(output_file), image)
                if verbose:
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
                
# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       type=str, default="../project/detpr/data/a2i2-haze/dry-run/2023/images", help="Image directory.")
    parser.add_argument("--label",       type=str, default="../project/detpr/run/predict/yolov8x-visdrone-a2i2-of-640/labels-voc", help="Bounding bbox directory.")
    parser.add_argument("--bbox-format", type=str, default="voc", help="Bounding bbox format: coco (xywh), voc (xyxy), yolo (cxcywhn).")
    parser.add_argument("--output",      type=str, default=None, help="Output directory.")
    parser.add_argument("--save-image",  default=True, action="store_true", help="Save image.")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = vars(parse_args())
    if args["bbox_format"] not in ["voc", "coco", "yolo"]:
        raise ValueError
    
    visualize_bboxes(args=args)
    
# endregion
