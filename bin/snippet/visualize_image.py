#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script visualizes images."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--image-dir", default=mon.DATA_DIR/"aic23-autocheckout/run/testA_1/visualize/", type=click.Path(exists=True), help="Image directory.")
@click.option("--verbose",   default=True, is_flag=True)
def visualize_image(
    image_dir: mon.Path,
    verbose  : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Visualizing"
        ):
            image = cv2.imread(str(image_files[i]))
            if verbose:
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord("q"):
                    break
                
# endregion


# region Main

if __name__ == "__main__":
    visualize_image()

# endregion
