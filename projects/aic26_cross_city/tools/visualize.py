#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize Bounding Boxes.

This script provides a CLI for visualizing bounding boxes.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import (
    BBoxes,
    BBoxFormat,
    Class,
    ClassList,
    Path,
    Size,
    create_progress_bar,
    resolve_project_root,
)
from mon.ops import load_bbox, read_image, write_image

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def visualize(args: argparse.Namespace):
    # Resolve paths
    root: Path = resolve_project_root(current_dir)
    data_dir = root / "data" / "aic26_cross_city" / args.src
    image_dir = data_dir / "image"
    label_dir = data_dir / args.label
    classes_file = data_dir / "classes.yaml"

    output_dir = data_dir / f"vis_{args.label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class-labels
    classes = ClassList.from_file(path=classes_file)

    # Retrieve images
    image_files: list[Path] = sorted(image_dir.glob("*"))
    image_files = [f for f in image_files if f.is_image_file(exists=True)]

    # Loop through each image
    with create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence=enumerate(image_files),
            total=len(image_files),
            description="[bright_yellow]Visualizing",
        ):
            # Load and convert bounding boxes to YOLO format
            label_file = label_dir / f"{image_file.stem}.txt"
            if not label_file.is_file():
                continue

            bboxes: BBoxes = load_bbox(
                path=label_file,
                fmt=BBoxFormat.CXCYWHN,
                image_file=image_file,
            )

            # Draw bbox on image
            image = read_image(path=image_file)
            imgsz = Size.from_any(image)

            for bbox in bboxes:
                label: Class = classes[bbox.class_id]
                image = bbox.draw(
                    image=image,
                    label=label.name,
                    imgsz=imgsz,
                    color=label.color,
                )

            # Save
            output_file = output_dir / image_file.name
            write_image(image=image, path=output_file)

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """The main function."""
    visualize(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--src",   type=str, default="x/eccv_cross_city", help="Data source directory.")
    parser.add_argument("--label", type=str, default="label",             help="Label directory.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
