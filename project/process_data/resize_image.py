#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script resize images."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--input-dir",    type=click.Path(exists=True),  default=mon.DATA_DIR, help="Image directory.")
@click.option("--output-dir",   type=click.Path(exists=False), default=None,         help="Output directory.")
@click.option("--imgsz",        type=int,                      default=512)
@click.option("--divisible-by", type=int,                      default=32)
@click.option("--side",         type=click.Choice(["short", "long", "vert", "horz"], case_sensitive=False), default="short")
@click.option("--replace",      is_flag=True)
@click.option("--verbose",      is_flag=True)
def resize_image(
    input_dir   : mon.Path,
    output_dir  : mon.Path,
    imgsz       : int,
    divisible_by: int,
    side        : bool,
    replace     : bool,
    verbose     : bool
):
    assert input_dir and mon.Path(input_dir).is_dir()
    input_dir   = mon.Path(input_dir)
    output_dir  = output_dir or input_dir.parent / f"{input_dir.stem}_resize"
    output_dir  = mon.Path(output_dir)
    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image   = cv2.imread(str(image_files[i]))
            resized = mon.resize(image, imgsz, divisible_by, side, cv2.INTER_AREA)
            if replace:
                cv2.imwrite(image_files[i], resized)
            else:
                output_file = output_dir / image_files[i].name
                mon.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(output_file, resized)
            
            if verbose:
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord("q"):
                    break

# endregion


# region Main

if __name__ == "__main__":
    resize_image()

# endregion
