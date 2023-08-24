#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script converts segmentation masks formats."""

from __future__ import annotations

import click
import cv2
import numpy as np

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Functions

@click.command()
@click.option("--image-dir",   default=mon.DATA_DIR/"aic23-autocheckout/train/tray/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--label-dir",   default=mon.DATA_DIR/"aic23-autocheckout/train/tray/labels-coco", type=click.Path(exists=True), help="Bounding bbox directory.")
@click.option("--from-format", default="voc",  type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--to-format",   default="yolo", type=click.Choice(["voc", "coco", "yolo"], case_sensitive=False), help="Bounding bbox format.")
@click.option("--output-dir",  default=None, type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",     is_flag=True)
def convert_segment(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_dir : mon.Path,
    from_format: str,
    to_format  : str,
    verbose    : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir  = mon.Path(image_dir)
    label_dir  = mon.Path(label_dir)
    output_dir = output_dir or image_dir.parent / f"labels-{to_format}"
    output_dir = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = mon.ShapeCode.from_value(value=f"{from_format}_to_{to_format}")
    
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
            
            src_label_file = label_dir  / f"{image_files[i].stem}.txt"
            dst_label_file = output_dir / f"{image_files[i].stem}.txt"
            
            # .txt file
            if src_label_file.is_txt_file():
                with open(src_label_file, "r") as in_file:
                    l = in_file.read().splitlines()
                l = [x.strip().split(" ") for x in l]
                l = [x for x in l if len(x) >= 2]
                s = [list(map(float, x[1:-1])) for x in l]
                
                with open(dst_label_file, "w") as out_file:
                    for j, x in enumerate(s):
                        x = np.array(x).reshape((-1, 2))
                        x = mon.convert_contour(contour=x, code=code, height=h, width=w)
                        out_file.write(f"{l[j][0]}")
                        for v in x:
                            out_file.write(f" {v}")
                        out_file.write("\n")
                    out_file.close()
            
            # .json file (COCO format)
            src_label_file = label_dir / f"{image_files[i].stem}.json"
            if src_label_file.is_json_file():
                data   = mon.read_from_file(path=src_label_file)
                shapes = data.get("shapes", None)
                if shapes is None:
                    continue
                with open(dst_label_file, "w") as out_file:
                    for j, s in enumerate(shapes):
                        l = s.get("label",  None)
                        x = s.get("points", None)
                        if x is None:
                            continue
                        x = np.array(x).reshape((-1, 1, 2))
                        x = mon.convert_contour(contour=x, code=code, height=h, width=w)
                        x = x.reshape(-1)
                        out_file.write(f"0")
                        for v in x:
                            out_file.write(f" {v}")
                        out_file.write("\n")
                    out_file.close()
                
# endregion


# region Main

if __name__ == "__main__":
    convert_segment()
    
# endregion
