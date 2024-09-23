#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualize bboxes in images.

python visualize_bbox.py --format "yolo" --verbose --image-dir "/home/longpham/10_workspace/11_code/mon/data/enhance/llie/darkface/test/lq" --label-dir "/home/longpham/10_workspace/11_code/mon/data/enhance/llie/darkface/test/labels"
"""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon


# region Function

@click.command()
@click.option("--image-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR, help="Image directory.")
@click.option("--label-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR, help="Image directory.")
@click.option("--output-dir", type=click.Path(exists=False), default=None,         help="Output directory.")
@click.option("--format",     type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), default="yolo", help="Bounding bbox format.")
@click.option("--verbose",    is_flag=True)
def convert_bbox(
    image_dir : mon.Path,
    label_dir : mon.Path,
    output_dir: mon.Path,
    format    : str,
    verbose   : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or label_dir.parent / f"{label_dir.stem}_yolo"
    output_dir  = mon.Path(output_dir)
    code        = mon.ShapeCode.from_value(value=f"{format}_to_voc")
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            
            label_file = label_dir / f"{image_files[i].stem}.txt"
            if label_file.is_txt_file():
                with open(label_file, "r") as in_file:
                    l = in_file.read().splitlines()
                l = [x.strip().split(" ") for x in l]
                l = [x for x in l if len(x) >= 2]
                b = np.array([list(map(float, x[0:])) for x in l])
                b = mon.convert_bbox(bbox=b, code=code, height=h, width=w)
            
            if verbose:
                for j, x in enumerate(b):
                    image = mon.draw_bbox(
                        image = image,
                        bbox  = x,
                    )
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord("q"):
                    break
        
# endregion


# region Main

if __name__ == "__main__":
    convert_bbox()

# endregion
