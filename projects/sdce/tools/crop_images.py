#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Crop Images.

This script provides a CLI for cropping all images in a given directory to the
same box defined by the user. The cropped images are saved in a new "cropped"
subdirectory within the input directory.

Example:

python crop_images.py --data "/Volumes/ssd_01/10_workspace/11_code/mon/projects/dev/run/assets/sice_112" --box "11800,600,200,200"
python crop_images.py --data "/Volumes/ssd_01/10_workspace/11_code/mon/projects/dev/run/assets/sice_229" --box "1100,1800,200,200"
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

import cv2

from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    # Validate inputs
    data_dir = Path(args.data).normalize()
    cropped_dir = data_dir / "cropped"
    highlighted_dir = data_dir / "highlighted"
    resized_dir = data_dir / "resized"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found at {data_dir.as_posix()}")

    cropped_dir.mkdir(parents=True, exist_ok=True)
    highlighted_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    box = [int(x) for x in args.box.split(",")]
    if len(box) != 4:
        raise ValueError(f"box must be in (top, left, height, width) format, "
                         f"got {args.box}.")

    # Get all images
    image_files = list(data_dir.glob("*"))
    image_files = [i for i in image_files if i.is_image_file(exists=True)]
    image_files.sort()

    # Process all images
    for image_file in image_files:
        image = cv2.imread(str(image_file))

        # Crop image
        cropped = image[box[0] : box[0] + box[2], box[1] : box[1] + box[3]]
        cropped_file = cropped_dir / image_file.name
        cv2.imwrite(str(cropped_file), cropped)

        # Highlight the cropped area on the original image
        highlighted = image.copy()
        cv2.rectangle(
            img=highlighted,
            pt1=(box[1], box[0]),
            pt2=(box[1] + box[3], box[0] + box[2]),
            color=(0, 0, 255),
            thickness=4,
        )
        highlighted_file = highlighted_dir / image_file.name
        cv2.imwrite(str(highlighted_file), highlighted)

        # Resize image (to reduce memory in the figure/paper)
        h, w, c = image.shape
        resized = cv2.resize(image, (w // args.scale, h // args.scale))
        cv2.rectangle(
            img=resized,
            pt1=(box[1] // args.scale, box[0] // args.scale),
            pt2=(box[1] // args.scale + box[3] // args.scale, box[0] // args.scale + box[2] // args.scale),
            color=(0, 0, 255),
            thickness=4,
        )
        resized_file = resized_dir / image_file.name
        cv2.imwrite(str(resized_file), resized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--data", type=str, default="/Volumes/ssd_01/10_workspace/11_code/mon/projects/dev/run/assets/sice_229")
    parser.add_argument("--box", type=str, default="1100,1800,200,200")
    parser.add_argument("--scale", type=int, default=8)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
