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
@click.option("--image-dir",   default=mon.DATA_DIR/"aic23-autocheckout/train/tray/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",   default=mon.DATA_DIR/"aic23-autocheckout/train/tray/labels-voc-1", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--from-format", default="voc", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--to-format",   default="yolo", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--output-dir",  default=mon.DATA_DIR/"aic23-autocheckout/train/tray/labels-voc", type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",     is_flag=True)
def convert_bbox(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_dir : mon.Path,
    from_format: str,
    to_format  : str,
    verbose    : bool
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
            prefix     = str(image_files[i].stem).split("_")
            prefix     = f"{prefix[0]}_{prefix[1]}_000000.txt"
            source     = label_dir / prefix
            label_file = output_dir / f"{image_files[i].stem}.txt"
            mon.copy_file(source, label_file)
            
# endregion


# region Main

if __name__ == "__main__":
    convert_bbox()
    
# endregion
