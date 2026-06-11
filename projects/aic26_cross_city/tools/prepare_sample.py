#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare Sample Dataset.

This script provides a CLI for preparing the sample dataset for training.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

import jsonlines
import numpy as np
from mon.core import (
    BBox,
    BBoxes,
    BBoxFormat,
    Path,
    Size,
    create_progress_bar,
    resolve_project_root,
)
from mon.ops import write_bbox

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def convert():
    """Convert the VOC format to YOLO format."""
    remap_ = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,  # person
    }

    # Resolve paths
    root: Path = resolve_project_root(current_dir)
    data_dir = root / "data" / "aic26_cross_city" / "x" / "eccv_cross_city"
    anns_file = data_dir / "annotations.jsonl"

    output_label_dir = data_dir / "label"
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each line
    with create_progress_bar() as pbar:
        with jsonlines.open(str(anns_file)) as f:
            for line in pbar.track(
                sequence=f,
                description="[bright_yellow]Processing",
            ):
                H = line["height"]
                W = line["width"]
                imgsz = Size(height=H, width=W)

                # Convert bboxes
                bboxes = []
                for b in line["bboxes"]:
                    x = b["top_left_x"]
                    y = b["top_left_y"]
                    w = b["width"]
                    h = b["height"]
                    cx = x + w / 2
                    cy = y + h / 2
                    bboxes.append(
                        BBox(
                            class_id=remap_[int(b["class_idx"])],
                            bbox=np.array([cx, cy, w, h]),
                            angle=0.0,
                            score=float(b["confidence"]),
                            track_id=-1,
                            imgsz=imgsz,
                        )
                    )
                bboxes = BBoxes(bboxes=bboxes, imgsz=imgsz)

                # Save
                stem = Path(line["remote_path"]).stem
                label_file = output_label_dir / f"{stem}.txt"
                write_bbox(bbox=bboxes, path=label_file, fmt=BBoxFormat.CXCYWHN)

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """The main function."""
    if args.convert:
        # Convert bbox format
        convert()
    else:
        print("No valid action specified. Use --convert.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--convert", action="store_true", help="Convert the bounding box.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
