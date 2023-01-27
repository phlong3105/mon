#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the visualization of bounding boxes on images."""

from __future__ import annotations

import argparse

import cv2
import munch
import numpy as np
import torch

import mon


# region Function

def visualize_bbox_coco(args: munch.Munch):
    """Visualize bounding boxes on images."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output
    output_dir = mon.Path(args.output) \
        if (output_dir is not None) \
        else (image_dir.parent / "draw")
    # mon.create_dirs(paths=[output_dir])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.progress_bar() as pbar:
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
                b = [list(map(float, x[1:])) for x in l]
                b = np.array(b)
                b = torch.from_numpy(b)
                b = mon.box_xywh_to_xyxy(b)
                b = b.numpy()
                
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
                

def visualize_bbox_voc(args: munch.Munch):
    """Visualize bounding boxes on images."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output
    output_dir = mon.Path(args.output) \
        if (output_dir is not None) \
        else (image_dir.parent / "draw")
    # mon.create_dirs(paths=[output_dir])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.progress_bar() as pbar:
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
                b = [list(map(float, x[1:])) for x in l]
                b = np.array(b)
                
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
                

def visualize_bbox_yolo(args: munch.Munch):
    """Visualize bounding boxes on images."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output
    output_dir = mon.Path(args.output) \
        if (output_dir is not None) \
        else (image_dir.parent / "draw")
    # mon.create_dirs(paths=[output_dir])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.progress_bar() as pbar:
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
                b = [list(map(float, x[1:])) for x in l]
                b = np.array(b)
                b = torch.from_numpy(b)
                b = mon.box_cxcywh_norm_to_xyxy(b, height=h, width=w)
                b = b.numpy()
                
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
        help    = "Bounding box directory."
    )
    parser.add_argument(
        "--box-format",
        type    = str,
        default = "yolo",
        help    = "Bounding box format (i.e., xyxy, cxcywh, cxcywh_norm (yolo)."
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
    args = munch.Munch.fromDict(vars(parse_args()))
    assert args.box_format in [
        "xyxy", "xywh", "cxcywh", "cxcywh_norm", "voc", "coco", "yolo",
    ]
    if args.box_format in ["xyxy", "voc"]:
        visualize_bbox_voc(args=args)
    elif args.box_format in ["xywh", "coco"]:
        visualize_bbox_coco(args=args)
    elif args.box_format in ["cxcywh"]:
        pass
    elif args.box_format in ["cxcywh_norm", "yolo"]:
        visualize_bbox_yolo(args=args)
    
# endregion
