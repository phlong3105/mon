#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare CityFlow Dataset.

This script provides a CLI for preparing the CityFlow dataset for training.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import BBoxes, BBoxFormat, Path, create_progress_bar, resolve_project_root
from mon.ops import load_bbox, write_bbox

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def convert():
    """Convert the VOC format to YOLO format."""
    remap_ = {
        "car": 0,
        "truck": 1,
    }

    # Resolve paths
    root: Path = resolve_project_root(current_dir)
    data_dir = root / "data" / "aic26_cross_city" / "x" / "cityflow" / "train"

    output_dir = data_dir.parent / "new"
    output_image_dir = output_dir / "image"
    output_bbox_dir = output_dir / "label"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_bbox_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each subdir
    subdirs = data_dir.subdirs()
    with create_progress_bar() as pbar:
        for subdir in pbar.track(
            sequence=subdirs,
            total=len(subdirs),
            description="[bright_yellow]Subdir",
        ):
            image_dir = subdir / "images"
            label_dir = subdir / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)

            image_files: list[Path] = sorted(image_dir.glob("*.jpeg"))

            # Process each image-object pair
            task = pbar.add_task(
                description="[bright_yellow]Processing",
                total=len(image_files),
            )
            for i, image_file in enumerate(image_files):
                if i % 20 != 0:
                    pbar.update(task, advance=1)
                    continue

                object_file = image_file.replace_part("/images/", "/objects/")
                object_file = object_file.xml_file

                # Load and convert bounding boxes to YOLO format
                bbox: BBoxes = load_bbox(
                    path=object_file,
                    fmt=BBoxFormat.XYXY,
                    remap=remap_,
                    image_file=image_file,
                )

                if bbox.is_empty:
                    pbar.update(task, advance=1)
                    continue

                # Save image-bbox to output directory
                output_image_file = output_image_dir / f"cityflow_{subdir.name}_{image_file.stem}.jpg"
                output_bbox_file = output_bbox_dir / f"cityflow_{subdir.name}_{image_file.stem}.txt"
                label_file = label_dir / f"{object_file.stem}.txt"

                write_bbox(bbox=bbox, path=output_bbox_file, fmt=BBoxFormat.CXCYWHN)
                image_file.copy_to(output_image_file)
                output_bbox_file.copy_to(label_file)

                pbar.update(task, advance=1)
            pbar.remove_task(task)


def remap():
    """Remap the class labels to a new set of labels."""
    remap_ = {
        0: 0,
        1: 4,
    }

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """The main function."""
    if args.convert:
        # Convert bbox format
        convert()
    elif args.remap:
        # Remap class IDs
        remap()
    else:
        print("No valid action specified. Use --convert or --remap.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--convert", action="store_true", help="Convert the bounding box.")
    parser.add_argument("--remap",   action="store_true", help="Remap class IDs.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
