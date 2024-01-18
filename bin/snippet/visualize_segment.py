#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes segmentation masks on images."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon


# region Function

@click.command()
@click.option("--image-dir",      default=mon.DATA_DIR / "aic23-autocheckout/run/testA_4/images/", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",      default=mon.DATA_DIR / "aic23-autocheckout/run/testA_4/labels/", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--output-dir",     default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--segment-format", default="yolo", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Segmentation mask format.")
@click.option("--thickness",      default=1, type=int, help="The thickness of the segmentation mask border line in px.")
@click.option("--bbox",           is_flag=True, help="Draw bounding box.")
@click.option("--label",          is_flag=True, help="Draw label.")
@click.option("--fill",           default=True, is_flag=True, help="Fill the region inside the segment with transparent color.")
@click.option("--point",          is_flag=True, help="Draw each point along the segment contour.")
@click.option("--radius",         default=3, type=int, help="The radius value of the point.")
@click.option("--extension",      default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--save",           is_flag=True)
@click.option("--verbose",        default=True, is_flag=True)
def visualize_segment(
    image_dir     : mon.Path,
    label_dir     : mon.Path,
    output_dir    : mon.Path,
    segment_format: str,
    thickness     : int,
    bbox          : bool,
    label         : bool,
    fill          : bool,
    point         : bool,
    radius        : int,
    extension     : bool,
    save          : bool,
    verbose       : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or label_dir.parent / "visualize"
    output_dir  = mon.Path(output_dir)
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    code = mon.ShapeCode.from_value(value=f"{segment_format}_to_voc")

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
           
            label_file = label_dir / f"{image_files[i].stem}.txt"
            if label_file.is_txt_file():
                with open(label_file, "r") as in_file:
                    l = in_file.read().splitlines()
                l      = [x.strip().split(" ") for x in l]
                l      = [x for x in l if len(x) >= 2]
                s      = [list(map(float, x[1:-1])) for x in l]
                colors = mon.RGB.values()
                n      = len(colors)
                for j, x in enumerate(s):
                    x     = np.array(x).reshape((-1, 2))
                    x     = mon.convert_contour(contour=x, code=code, height=h, width=w)
                    image = mon.draw_segment(
                        image     = image,
                        segment   = x,
                        bbox      = bbox,
                        label     = l[j] if label else None,
                        color     = colors[abs(hash(l[j][0])) % n],
                        thickness = thickness,
                        fill      = fill,
                        point     = point,
                        radius    = radius
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
            if save:
                output_file = output_dir / f"{image_files[i].stem}.{extension}"
                cv2.imwrite(str(output_file), image)
            if verbose:
                cv2.imshow("Image", image)
                cv2.waitKey(0)
                
# endregion


# region Main

if __name__ == "__main__":
    visualize_segment()

# endregion
