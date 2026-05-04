#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Copy Images.

This script provides a CLI for copying input and target images from a dataset
and the corresponding predicted images from all methods to the same directory
for further analysis.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon import Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    # Validate inputs
    dataset = args.dataset
    filename = args.filename

    data_dir = Path(args.data_dir).normalize()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found at {data_dir.as_posix()}")

    output_dir = Path(args.output_dir).normalize()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output directory not found at {output_dir.as_posix()}")

    # Loop through all method directories in the data directory
    for image_file in data_dir.rglob(f"*/{dataset}/*/{filename}"):
        if not image_file.is_image_file(exists=True):
            continue

        method_name = image_file.parents[2].name
        output_file = output_dir / f"{dataset}_{image_file.stem}" / f"{method_name}.jpg"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.copy_to(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--data-dir", type=str, default="/Volumes/ssd_01/10_workspace/11_code/mon/projects/dev/run/predict/")
    parser.add_argument("--output-dir", type=str, default="/Volumes/ssd_01/10_workspace/11_code/mon/projects/dev/run/assets/")
    parser.add_argument("--dataset", type=str, default="sice")
    parser.add_argument("--filename", type=str, default="229.jpg")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
