#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the visualization of bounding boxes on images."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon


# region Function

@click.command()
@click.option("--image-dir",   default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",   default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/labels-voc", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--bbox-format", default="voc", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--output-dir",  default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--save-image",  is_flag=True)
@click.option("--verbose",     is_flag=True)
def visualize_bbox(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_dir : mon.Path,
    bbox_format: str,
    save_image : bool,
    verbose    : bool
):
    """Visualize bounding boxes on images."""
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or label_dir.parent / "visualize"
    output_dir  = mon.Path(output_dir)
    if save_image:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
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
                
                if bbox_format in ["coco"]:
                    b = mon.bbox_xywh_to_xyxy(bbox=b)
                elif bbox_format in ["yolo"]:
                    b = mon.bbox_cxcywhn_to_xyxy(bbox=b, height=h, width=w)
                
                colors = mon.RGB.values()
                n      = len(colors)
                for j, x in enumerate(b):
                    mon.draw_bbox(
                        image     = image,
                        bbox      = x,
                        color     = colors[abs(hash(l[j][0])) % n],
                        thickness = 2,
                    )
                
                image = cv2.putText(
                    img       = image,
                    text      = f"{image_files[i].stem}",
                    org       = [50, 50],
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1,
                    color     = [255, 255, 255],
                    thickness = 2,
                    lineType  = cv2.LINE_AA,
                )
                if save_image:
                    output_file = output_dir / f"{image_files[i].name}"
                    cv2.imwrite(str(output_file), image)
                if verbose:
                    cv2.imshow("Image", image)
                    cv2.waitKey(0)
                
# endregion


# region Main

if __name__ == "__main__":
    visualize_bbox()

# endregion
