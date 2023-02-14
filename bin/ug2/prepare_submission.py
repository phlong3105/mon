#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the script to convert results to UG2+ submission
format.
"""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",  default=mon.DATA_DIR / "a2i2-haze/dry-run/2023/labels-voc", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--output-dir", default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--conf",       default=0.25, type=float, help="Object confidence threshold for detection.")
def prepare_submission(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_dir : mon.Path,
    conf       : float
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
        
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_dir  = output_dir or image_dir.parent / "labels-voc"
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
                b = mon.bbox_cxcywhn_to_xyxy(bbox=b, height=h, width=w)
                
            with open(label_file_yolo, "x") as out_file:
                for j, x in enumerate(b):
                    if float(l[j][5]) < conf:
                        continue
                    out_file.write(
                        "vehicle "
                        f"{x[0]:.02f} "
                        f"{x[1]:.02f} "
                        f"{x[2]:.02f} "
                        f"{x[3]:.02f} "
                        f"{float(l[j][5]):.02f}\n"
                    )
                out_file.close()

# endregion


# region Main

if __name__ == "__main__":
    prepare_submission()

# endregion
