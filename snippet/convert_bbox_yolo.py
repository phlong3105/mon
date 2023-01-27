#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts bounding boxes to YOLO's format."""

from __future__ import annotations

import argparse

import cv2
import munch
import numpy as np
import torch

import mon


# region Functions

def convert_bbox_coco_to_yolo(args: munch.Munch):
    """Convert bounding box to YOLO format."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output
    output_dir = mon.Path(args.output) \
        if (output_dir is not None) \
        else (image_dir.parent / "labels-yolo")
    mon.create_dirs(paths=[output_dir])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.progress_bar() as pbar:
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
                b = [list(map(float, x[1:])) for x in l]
                b = np.array(b)
                b = torch.from_numpy(b)
                b = mon.box_xywh_to_cxcywh_norm(b, height=h, width=w)
                b = b.numpy()
            with open(label_file_yolo, "w") as out_file:
                for j, x in enumerate(b):
                    out_file.write(f"{l[j][0]} {x[1]} {x[2]} {x[3]} {x[4]}\n")
                out_file.close()


def convert_bbox_voc_to_yolo(args: munch.Munch):
    """Convert bounding box to YOLO format."""
    assert args.image is not None and mon.Path(args.image).is_dir()
    assert args.label is not None and mon.Path(args.label).is_dir()
    image_dir  = mon.Path(args.image)
    label_dir  = mon.Path(args.label)
    output_dir = args.output
    output_dir = mon.Path(args.output) \
        if (output_dir is not None) \
        else (image_dir.parent / "labels-yolo")
    mon.create_dirs(paths=[output_dir])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.rich.progress_bar() as pbar:
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
                b = [list(map(float, x[1:])) for x in l]
                b = np.array(b)
                b = torch.from_numpy(b)
                b = mon.box_xyxy_to_cxcywh_norm(b, height=h, width=w)
                b = b.numpy()
            with open(label_file_yolo, "w") as out_file:
                for j, x in enumerate(b):
                    out_file.write(f"{l[j][0]} {x[0]} {x[1]} {x[2]} {x[3]}\n")
                out_file.close()

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/dry-run/2023/images",
        help    = "Image directory."
    )
    parser.add_argument(
        "--label",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/dry-run/2023/labels-voc",
        help    = "Bounding box directory."
    )
    parser.add_argument(
        "--box-format",
        type    = str,
        default = "voc",
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
        convert_bbox_voc_to_yolo(args=args)
    elif args.box_format in ["xywh", "coco"]:
        convert_bbox_coco_to_yolo(args=args)
    elif args.box_format in ["cxcywh"]:
        pass
    elif args.box_format in ["cxcywh_norm", "yolo"]:
        pass
    
# endregion
