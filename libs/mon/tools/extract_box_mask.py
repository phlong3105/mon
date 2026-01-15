#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Box mask extraction tool.

This module provides a tool for extracting masks from images using bounding box
annotations and a segmentation model.
"""

from __future__ import annotations

__all__ = []

import argparse

import mon
from mon.cv import segment

current_file = mon.Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]
data_dir     = current_dir.data_dir()


# ==============================================================================
# region CONTROL
# ==============================================================================

def run(args: argparse.Namespace):
    """Run the box mask extraction process."""
    # Resolve paths
    image_dirs = (data_dir / args.data).image_dirs(recursive=True)

    # Initialize extractor
    segmentor = segment.SAMBoxSegmentor(
        name    = args.model,
        device  = args.device,
        verbose = args.verbose
    )

    # Loop over image directories
    for image_dir in image_dirs:
        label_dir  = image_dir.parent / "label"
        output_dir = image_dir.parent / "mask"

        # Skip if the label directory does not exist
        if label_dir.exists():
            segmentor.process_dir(image_dir=image_dir, label_dir=label_dir, output_dir=output_dir)

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, required=True, help="Source data name inside 'data' directory.")
    parser.add_argument("--model",   type=str, default="sam2.1_b")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()

# endregion
