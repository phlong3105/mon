#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module converts bounding boxes to YOLO's format."""

from __future__ import annotations

import argparse

import cv2
import munch
import torch

from mon import core, coreimage as ci


# region Functions

def convert_bbox(args: dict | munch.Munch):
    """Convert bounding box to YOLO format."""
    assert args.src is not None and core.Path(args.src).is_dir()
    labels_dir      = core.Path(args.src)
    images_dir      = labels_dir.parent / "images"
    assert images_dir.is_dir() and labels_dir.is_dir()
    labels_yolo_dir = args.dst
    labels_yolo_dir = core.Path(labels_yolo_dir) \
        if (labels_yolo_dir is not None) \
        else (labels_dir.parent / "labels-yolo")
    core.create_dirs(paths=[labels_yolo_dir])

    image_files = list(images_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    with core.rich.progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            
            label_file      = labels_dir      / f"{image_files[i].stem}.txt"
            label_yolo_file = labels_yolo_dir / f"{image_files[i].stem}.txt"
            with open(label_file, "r") as in_file:
                lines = in_file.read().splitlines()
            with open(label_yolo_file, "w") as out_file:
                for l in lines:
                    d   = l.split(" ")
                    if len(d) <= 5:
                        continue
                    box = torch.Tensor([[float(d[1]), float(d[2]), float(d[3]), float(d[4])]])
                    box = ci.box_xyxy_to_cxcywh_norm(box, height=h, width=w)
                    box = box[0]
                    out_file.write(
                        f"{d[0]} {box[0]} {box[1]} {box[2]} {box[3]}\n"
                    )
                out_file.close()
                
# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src"    , type   = str         , default = core.DATA_DIR / "a2i2-haze/test/labels", help = "Input source.")
    parser.add_argument("--dst"    , type   = str         , default = None                                   , help = "Output destination.")
    parser.add_argument("--verbose", action = "store_true", help    = "Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    input_args = vars(parse_args())
    src        = input_args.get("src",     None)
    dst        = input_args.get("dst",     None)
    verbose    = input_args.get("verbose", False)
    args = munch.Munch(
        src     = src,
        dst     = dst,
        verbose = verbose,
    )
    convert_bbox(args=args)

# endregion
