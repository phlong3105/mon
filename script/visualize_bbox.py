#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the visualization of bounding boxes on images."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import mon


# region Function

def visualize_bboxes(args: argparse.Namespace):
    """Visualize bounding boxes on images."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output or image_dir.parent / "draw"
    output_dir = mon.Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.get_progress_bar() as pbar:
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
                
                if args.from_format in ["coco"] and args.to_format in ["voc"]:
                    b = mon.bbox_xywh_to_xyxy(bbox=b)
                elif args.from_format in ["yolo"] and args.to_format in ["voc"]:
                    b = mon.bbox_cxcywhn_to_xyxy(bbox=b, height=h, width=w)
                    
                org        = (50, 50)
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness  = 2
                colors     = mon.RGB.values()
                n          = len(colors)
                for j, x in enumerate(b):
                    start = (int(x[0]), int(x[1]))
                    end   = (int(x[2]), int(x[3]))
                    color = colors[abs(hash(l[j][0])) % n]
                    image = cv2.rectangle(image, start, end, color, thickness)

                image = cv2.putText(
                    image, f"{image_files[i].stem}", org, font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA,
                )
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                
# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/train/detection/haze/images",
        help    = "Image directory."
    )
    parser.add_argument(
        "--label",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/train/detection/haze/labels",
        help    = "Bounding bbox directory."
    )
    parser.add_argument(
        "--bbox-format",
        type    = str,
        default = "yolo",
        help    = "Bounding bbox format: coco (xywh), voc (xyxy), yolo (cxcywhn)."
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = None,
        help    = "Output directory."
    )
    parser.add_argument("--verbose", action = "store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.bbox_format not in ["voc", "coco", "yolo"]:
        raise ValueError

    visualize_bboxes(args=args)
    
# endregion
