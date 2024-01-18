#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script creates bounding boxes' labels for all images."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Functions

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"aic23-autocheckout/train/testA-02/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",  default=mon.DATA_DIR/"aic23-autocheckout/train/testA-02/labels-yolo", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--output-dir", default=mon.DATA_DIR/"aic23-autocheckout/train/testA-02/labels-yolo2", type=click.Path(exists=False), help="Output directory.")
def convert_bbox(
    image_dir : mon.Path,
    label_dir : mon.Path,
    output_dir: mon.Path,
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or image_dir.parent / "labels-yolo"
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
            if i % 4 != 0:
                continue
            stem           = str(image_files[i].stem)
            label_file     = label_dir  / f"{stem}.txt"
            output_file    = output_dir / f"{stem}.txt"
            new_image_file = image_dir.parent  / f"{image_dir.name}_2" / image_files[i].name
            new_image_file.parent.mkdir(parents=True, exist_ok=True)
            if label_file.is_txt_file(exist=True):
                mon.copy_file(label_file, output_file)
                mon.copy_file(image_files[i], new_image_file)
            
            
# endregion


# region Main

if __name__ == "__main__":
    convert_bbox()
    
# endregion
