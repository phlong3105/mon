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
@click.option("--mask-dir", default=mon.DATA_DIR/"aic23-autocheckout/testA/testA_1/person/", type=click.Path(exists=True), help="Mask directory.")
@click.option("--dilate",   default=5, type=int, help="Dilation size.")
@click.option("--verbose",  is_flag=True)
def postprocess_mask(
    mask_dir: mon.Path,
    dilate  : int,
    verbose : bool
):
    """Visualize bounding boxes on images."""
    assert mask_dir is not None and mon.Path(mask_dir).is_dir()
    
    mask_dir   = mon.Path(mask_dir)
    mask_files = list(mask_dir.rglob("*"))
    mask_files = [m for m in mask_files if m.is_image_file()]
    mask_files = sorted(mask_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(mask_files)),
            total       = len(mask_files),
            description = f"[bright_yellow] Processing"
        ):
            mask_file = mask_files[i]
            mask      = cv2.imread(str(mask_file))
            if dilate > 0:
                kernel = np.ones((dilate, dilate), np.uint8)
                mask   = cv2.dilate(mask, kernel, iterations=1)
            cv2.imwrite(str(mask_file), mask)
            if verbose:
                cv2.imshow("Mask", mask)
                if cv2.waitKey(1) == ord("q"):
                    break
                
# endregion


# region Main

if __name__ == "__main__":
    postprocess_mask()

# endregion
