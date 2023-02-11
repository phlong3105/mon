#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts bounding boxes to YOLO's format."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import mon


# region Functions

def convert_bboxes(args: dict):
    """Convert bounding boxes"""
    assert args["image"] is not None and mon.Path(args["image"]).is_dir()
    assert args["label"] is not None and mon.Path(args["label"]).is_dir()
    
    from_format = args["from_format"]
    to_format   = args["to_format"]
    
    image_dir   = mon.Path(args["image"])
    label_dir   = mon.Path(args["label"])
    output_dir  = args["output"] or image_dir.parent / "labels-yolo"
    output_dir  = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image           = cv2.imread(str(image_files[i]))
            h, w, c         = image.shape
            label_file      = label_dir  / f"{image_files[i].stem}.txt"
            label_file_yolo = output_dir / f"{image_files[i].stem}.txt"
            with open(label_file, "r") as in_file:
                l = in_file.read().splitlines()
                l = [x.strip().split(" ") for x in l]
                l = [x for x in l if len(x) >= 5]
                b = np.array([list(map(float, x[1:])) for x in l])
                
                if from_format in ["voc"] and to_format in ["coco"]:
                    b = mon.bbox_xyxy_to_xywh(bbox=b)
                elif from_format in ["voc"] and to_format in ["yolo"]:
                    b = mon.bbox_xyxy_to_cxcywhn(bbox=b, height=h, width=w)
                elif from_format in ["coco"] and to_format in ["voc"]:
                    b = mon.bbox_xywh_to_xyxy(bbox=b)
                elif from_format in ["coco"] and to_format in ["yolo"]:
                    b = mon.bbox_xywh_to_cxcywhn(bbox=b, height=h, width=w)
                elif from_format in ["yolo"] and to_format in ["voc"]:
                    b = mon.bbox_cxcywhn_to_xyxy(bbox=b, height=h, width=w)
                elif from_format in ["yolo"] and to_format in ["coco"]:
                    b = mon.bbox_cxcywhn_to_xywh(bbox=b, height=h, width=w)
                   
            with open(label_file_yolo, "w") as out_file:
                for j, x in enumerate(b):
                    out_file.write(f"{l[j][0]} {x[0]} {x[1]} {x[2]} {x[3]}\n")
                out_file.close()

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       type=str, default=mon.DATA_DIR/"a2i2-haze/dry-run/2023/images", help="Image directory.")
    parser.add_argument("--label",       type=str, default=mon.DATA_DIR/"a2i2-haze/dry-run/2023/labels-voc", help="Bounding bbox directory.")
    parser.add_argument("--from-format", type=str, default="voc", help="Bounding bbox format: coco (xywh), voc (xyxy), yolo (cxcywhn).")
    parser.add_argument("--to-format",   type=str, default="yolo", help="Bounding bbox format: coco (xywh), voc (xyxy), yolo (cxcywhn).")
    parser.add_argument("--output",      type=str, default=None, help="Output directory.")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = vars(parse_args())
    if args["from_format"] not in ["voc", "coco", "yolo"]:
        raise ValueError
    if args["to_format"] not in ["voc", "coco", "yolo"]:
        raise ValueError
    
    if args["from_format"] == args["to_format"]:
        pass
    else:
        convert_bboxes(args=args)
    
# endregion
