#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script crops images."""

from __future__ import annotations

import click
import cv2

import mon


_INCLUDE_DIRS = [
    "dicm",
    "fusion",
    "lime",
    "lol-v1",
    "lol-v2-real",
    "lol-v2-syn",
    "mef",
    "npe",
    "vv",
]

_EXCLUDE_DIRS = [
    "darkcityscapes",
    "darkface",
    "fivek-e",
]


# region Function

@click.command()
@click.option("--input-dir",  type=click.Path(exists=True),  default=mon.DATA_DIR/"llie/predict",      help="Image directory.")
@click.option("--output-dir", type=click.Path(exists=False), default=mon.DATA_DIR/"llie/predict-crop", help="Output directory.")
@click.option("--extension",  type=click.Choice(["jpg", "png"], case_sensitive=False),  default="png", help="Image extension.")
@click.option("--verbose",    is_flag=True)
def visualize_image(
    input_dir : mon.Path,
    output_dir: mon.Path,
    extension : str,
    verbose   : bool
):
    assert input_dir and mon.Path(input_dir).is_dir()

    output_dir  = output_dir or input_dir.parent / f"{input_dir.stem}-crop"
    output_dir  = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            skip = any(d in str(image_files[i]) for d in _EXCLUDE_DIRS) or \
                   all(d not in str(image_files[i]) for d in _INCLUDE_DIRS)
            if skip:
                continue

            image   = cv2.imread(str(image_files[i]))
            h, w, c = image.shape
            s       = min(h, w)
            x       = int(w / 2 - s / 2)
            y       = int(h / 2 - s / 2)
            crop    = image[y:y + s, x:x + s]

            # output_file = output_dir / f"{image_files[i].stem}.{extension}"
            output_file = image_files[i]
            output_file = output_file.replace("predict", "predict-crop")
            mon.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_file, crop)

            if verbose:
                cv2.imshow("Image", image)
                if cv2.waitKey(1) == ord("q"):
                   break

# endregion


# region Main

if __name__ == "__main__":
    visualize_image()

# endregion
