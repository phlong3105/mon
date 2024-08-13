#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate YOLO bounding boxes format."""

from __future__ import annotations

import click
import cv2

import mon
from mon.core.file import json


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"aic23-autocheckout/train/synthetic-00/images", type=click.Path(exists=True),  help="Image directory.")
@click.option("--output-dir", default=mon.DATA_DIR/"aic23-autocheckout/train/synthetic-00/labels", type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",    is_flag=True)
def gen_yolo_label(
    image_dir : mon.Path,
    output_dir: mon.Path,
    verbose   : bool
):
    assert image_dirImageDataset and mon.Path(image_dir).is_dir()

    image_dir   = mon.Path(image_dir)
    output_dir  = output_dir or image_dir.parent / f"labels"
    output_dir  = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)

    with mon.get_progress_bar() as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Extracting"
        ):
            image       = cv2.imread(str(image_file))
            h, w, c     = image.shape
            stem        = image_file.stem
            cls_id      = int(stem.split("_")[0])
            cx          = (0 + (w / 2.0)) / w
            cy          = (0 + (h / 2.0)) / h
            w           = w / w
            h           = h / h
            output_file = output_dir / f"{image_file.stem}.txt"
            
            with open(output_file, "w") as out:
                out.write(f"{cls_id} {cx} {cy} {w} {h} \n")

# endregion


# region Main

if __name__ == "__main__":
    gen_yolo_label()

# endregion
