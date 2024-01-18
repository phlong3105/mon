#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script splits images into two smaller patches."""

from __future__ import annotations

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--source",      default=mon.DATA_DIR/"a2i2-haze/train/dehazing/paired-haze-hazefree-images", type=click.Path(exists=True), help="Image directory.")
@click.option("--destination", default=mon.DATA_DIR/"a2i2-haze/train/dehazing/split", type=click.Path(exists=False), help="Output directory.")
@click.option("--mode",        default="horizontal", type=click.Choice(["horizontal", "hor", "h", "vertical", "ver", "v"], case_sensitive=False), help="Image directory.")
@click.option("--extension",   default="jpg", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",     is_flag=True)
def split_image(
    source     : mon.Path,
    destination: mon.Path,
    mode       : str,
    extension  : str,
    verbose    : bool
):
    assert source is not None and mon.Path(source).is_dir()
    
    source      = mon.Path(source)
    image_files = list(source.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)

    destination = destination or source.parent / "split"
    destination = mon.Path(destination)
    
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Visualizing"
        ):
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            
            if mode in ["horizontal", "hor", "h"]:
                half  = h // 2
                image1 = image[:half, :]
                image2 = image[half:, :]
            elif mode in ["vertical", "ver", "v"]:
                half   = w // 2
                image1 = image[:, :half]
                image2 = image[:, half:]
            
            image1_file = destination / "part1" / f"{image_files[i].stem}.{extension}"
            image2_file = destination / "part2" / f"{image_files[i].stem}.{extension}"
            image1_file.parent.mkdir(parents=True, exist_ok=True)
            image2_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image1_file), image1)
            cv2.imwrite(str(image2_file), image2)
            
            if verbose:
                cv2.imshow("Image1", image1)
                cv2.imshow("Image2", image2)
                if cv2.waitKey(1) == ord("q"):
                    break
                
# endregion


# region Main

if __name__ == "__main__":
    split_image()

# endregion
