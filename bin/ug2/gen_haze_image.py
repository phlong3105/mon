#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the script to generate hazy images for training object
detection models.
"""

from __future__ import annotations

import argparse

import albumentations as A
import cv2

import mon

_current_dir = mon.Path(__file__).absolute().parent


# region Function

def main(args: dict):
    assert args["image"] is not None and mon.Path(args["image"]).is_dir()
    
    verbose    = args["verbose"]
    
    image_dir  = mon.Path(args["image"])
    output_dir = args["output"] or image_dir.parent / f"{image_dir.stem}-haze"
    output_dir = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = A.Compose([
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.3, alpha_coef=0.7, p=1),
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   default=mon.DATA_DIR / "a2i2-haze/train/detection/hazefree/images", help="Directory for images or video.")
    parser.add_argument("--output",  default=mon.DATA_DIR / "a2i2-haze/train/detection/hazefree/images-haze", help="Output directory.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = vars(parse_args())
    main(args=args)

# endregion
