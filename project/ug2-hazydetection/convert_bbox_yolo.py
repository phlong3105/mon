#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module converts bounding boxes to YOLO's format."""

from __future__ import annotations

import argparse

import cv2
import munch
import torch

import mon


# region Functions

def convert_bbox(args: munch.Munch):
    """Convert bounding box to YOLO format."""
    assert args.src is not None and mon.Path(args.src).is_dir()
    label_dir      = mon.Path(args.src)
    image_dir      = label_dir.parent / "images"
    assert image_dir.is_dir() and label_dir.is_dir()
    label_yolo_dir = args.dst
    label_yolo_dir = mon.Path(label_yolo_dir) \
        if (label_yolo_dir is not None) \
        else (label_dir.parent / "labels-yolo")
    mon.create_dirs(paths=[label_yolo_dir])

    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    with mon.rich.progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            
            label_file      = label_dir      / f"{image_files[i].stem}.txt"
            label_yolo_file = label_yolo_dir / f"{image_files[i].stem}.txt"
            with open(label_file, "r") as in_file:
                lines = in_file.read().splitlines()
            with open(label_yolo_file, "w") as out_file:
                for l in lines:
                    d   = l.split(" ")
                    if len(d) <= 5:
                        continue
                    box = torch.Tensor([[float(d[1]), float(d[2]), float(d[3]), float(d[4])]])
                    box = mon.box_xyxy_to_cxcywh_norm(box, height=h, width=w)
                    box = box[0]
                    out_file.write(
                        f"{d[0]} {box[0]} {box[1]} {box[2]} {box[3]}\n"
                    )
                out_file.close()
                
# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src"    , type   = str         , default = mon.DATA_DIR / "a2i2-haze/test/labels", help = "Input source.")
    parser.add_argument("--dst"    , type   = str         , default = None                                  , help = "Output destination.")
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
