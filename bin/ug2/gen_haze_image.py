#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the script to generate hazy images for training object
detection models.
"""

from __future__ import annotations

import albumentations as A
import click
import cv2

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"a2i2-haze/train/detection/hazesynthetic01/images", type=click.Path(exists=True), help="Image directory.")
@click.option("--output-dir", default=mon.DATA_DIR/"a2i2-haze/train/detection/hazesynthetic01/images-haze", type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",    is_flag=True)
def gen_haze_image(
    image_dir : mon.Path,
    output_dir: mon.Path,
    verbose   : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    
    image_dir  = mon.Path(image_dir)
    output_dir = output_dir or image_dir.parent / f"{image_dir.stem}-haze"
    output_dir = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = A.Compose([
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.7, p=1),
        # A.RandomBrightnessContrast()
    ])
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image = cv2.imread(str(image_files[i]))
            haze  = transform(image=image)["image"]
            
            output_file = output_dir / f"{image_files[i].name}"
            cv2.imwrite(str(output_file), haze)
            # if verbose:
            cv2.imshow("Image", image)
            cv2.imshow("Haze",  haze)
            if verbose:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
    
# endregion


# region Main

if __name__ == "__main__":
    gen_haze_image()

# endregion
