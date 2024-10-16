#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Copy Image.

This script is used to process images for paper.
"""

from __future__ import annotations

import click
import cv2

import mon

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Main

@click.command(name="copy", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--input-dir",    type=click.Path(exists=True),  default=current_dir/"run/predict", help="Input directory.")
@click.option("--output-dir",   type=click.Path(exists=False), default=current_dir/"run/paper",   help="Output directory.")
@click.option("--image-file",   type=click.Path(exists=False), default=None, help="e.g., 'dataset/image_id.jpg'")
@click.option("--imgsz",        type=int,                      default=512)
@click.option("--divisible-by", type=int,                      default=32)
@click.option("--side",         type=click.Choice(["short", "long", "vert", "horz"], case_sensitive=False), default="short")
@click.option("--resize",       is_flag=True)
@click.option("--verbose",      is_flag=True)
def main(
    input_dir   : mon.Path,
    output_dir  : mon.Path,
    image_file  : mon.Path,
    imgsz       : int,
    divisible_by: int,
    side        : bool,
    resize      : bool,
    verbose     : bool
):
    assert input_dir and mon.Path(input_dir).is_dir()
    image_file = mon.Path(image_file)
    image_file = image_file.parent / image_file.stem
    input_dir  = mon.Path(input_dir)
    output_dir = output_dir or input_dir.parent / f"{input_dir.stem}_copy"
    output_dir = mon.Path(output_dir)
    output_dir = output_dir / image_file
    #
    image_files = list(input_dir.rglob(f"*/{image_file}.*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Copying"
        ):
            path  = mon.Path(image_files[i])
            image = cv2.imread(str(image_files[i]))
            if resize:
                image = mon.resize(image, imgsz, divisible_by, side, cv2.INTER_AREA)
            
            data_name   = path.parents[0].name
            model_name  = path.parents[1].name
            arch_name   = path.parents[2].name
            output_file = output_dir / f"{model_name}{path.suffix}"
            mon.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_file, image)
            

if __name__ == "__main__":
    main()

# endregion
