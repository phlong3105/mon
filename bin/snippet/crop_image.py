#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script crops images."""

from __future__ import annotations

import click
import cv2

import mon
import mon.vision.core.image


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"llie/predict", type=click.Path(exists=True),  help="Image directory.")
@click.option("--output-dir", default=mon.DATA_DIR/"llie/predict-crop", type=click.Path(exists=False), help="Output directory.")
@click.option("--extension",  default="png", type=click.Choice(["jpg", "png"], case_sensitive=False), help="Image extension.")
@click.option("--verbose",    is_flag=True)
def visualize_image(
    image_dir  : mon.Path,
    output_dir : mon.Path,
    extension  : str,
    verbose    : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()

    output_dir  = output_dir or image_dir.parent / f"{image_dir.stem}-crop"
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
            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            s       = min(h, w)
            cy, cx  = h / 2, w/2
            x       = cx - s / 2
            y       = cy - s / 2
            crop    = image[y:y + s, x:x + s]

            result_file = output_dir / f"{image_files[i].stem}.{extension}"
            cv2.imwrite(str(result_file), crop)

            if verbose:
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord("q"):
                   break

# endregion


# region Main

if __name__ == "__main__":
    visualize_image()

# endregion
