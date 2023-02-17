#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes segmentation masks on images."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

console = mon.console


# region Function

@click.command()
@click.option("--image-dir",      default=mon.DATA_DIR/"aic23-checkout/testA/testA_4/images/", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",      default=mon.DATA_DIR/"aic23-checkout/testA/testA_4/person/labels/", type=click.Path(exists=True), help="Segmentation label directory.")
@click.option("--output-dir",     default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--segment-format", default="yolo", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Segmentation mask format.")
@click.option("--dilate",         default=5, type=int, help="Dilation size.")
@click.option("--thickness",      default=1, type=int, help="The thickness of the segmentation mask border line in px.")
@click.option("--bbox",           is_flag=True, help="Draw bounding box.")
@click.option("--label",          is_flag=True, help="Draw label.")
@click.option("--fill",           default=True, is_flag=True, help="Fill the region inside the segment with transparent color.")
@click.option("--point",          is_flag=True, help="Draw each point along the segment contour.")
@click.option("--radius",         default=3, type=int, help="The radius value of the point.")
@click.option("--extension",      default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--save",           is_flag=True)
@click.option("--verbose",        is_flag=True)
def gen_human_mask(
    image_dir     : mon.Path,
    label_dir     : mon.Path,
    output_dir    : mon.Path,
    segment_format: str,
    dilate        : int,
    thickness     : int,
    bbox          : bool,
    label         : bool,
    fill          : bool,
    point         : bool,
    radius        : int,
    extension     : str,
    save          : bool,
    verbose       : bool
):
    """Visualize bounding boxes on images."""
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or label_dir.parent / "person-masks"
    output_dir  = mon.Path(output_dir)
    if save:
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
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            
            label_file = label_dir / f"{image_files[i].stem}.jpg"
            mask       = np.zeros_like(image)
            if label_file.is_image_file():
                mask = cv2.imread(str(label_file))
            """
            label_file = label_dir / f"{image_files[i].stem}.txt"
            if label_file.is_txt_file():
                with open(label_file, "r") as in_file:
                    l = in_file.read().splitlines()
                l = [x.strip().split(" ") for x in l]
                l = [x for x in l if len(x) >= 5]
                s = [list(map(float, x[1:-1])) for x in l]
                
                for j, segment in enumerate(s):
                    segment = np.array(segment).reshape((-1, 2))
                    code    = mon.ShapeCode.from_value(value=f"{segment_format}_to_voc")
                    segment = mon.convert_contour(contour=segment, code=code, height=h, width=w)
                    mask    = mon.draw_segment(
                        image     = mask,
                        segment   = segment,
                        bbox      = bbox,
                        label     = l[j] if label else None,
                        color     = [255, 255, 255],
                        thickness = thickness,
                        fill      = fill,
                        point     = point,
                        radius    = radius,
                    )
           """
            if dilate > 0:
                kernel = np.ones((dilate, dilate), np.uint8)
                mask   = cv2.dilate(mask, kernel, iterations=1)
            if save:
                output_image_file = output_dir / f"{image_files[i].stem}.{extension}"
                output_mask_file  = output_dir / f"{image_files[i].stem}_mask001.{extension}"
                cv2.imwrite(str(output_image_file), image)
                cv2.imwrite(str(output_mask_file),  mask)
            if verbose:
                cv2.imshow("Image", image)
                cv2.imshow("Mask",  mask)
                if cv2.waitKey(1) == ord("q"):
                    break
                
# endregion


# region Main

if __name__ == "__main__":
    gen_human_mask()

# endregion
