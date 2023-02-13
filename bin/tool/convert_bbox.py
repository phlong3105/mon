#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts bounding boxes to YOLO's format."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Functions

@click.command()
@click.option("--image-dir",   default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",   default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/labels-voc", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--from-format", default="voc", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--to-format",   default="yolo", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--output-dir",  default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",     is_flag=True)
def convert_bbox(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_dir : mon.Path,
    from_format: str,
    to_format  : str,
    verbose    : bool
):
    """Convert bounding boxes"""
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

if __name__ == "__main__":
    convert_bbox()
    
# endregion
