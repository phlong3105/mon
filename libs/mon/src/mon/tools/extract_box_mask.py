#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Box mask extractor.

This module provides a runner for extracting box masks from images.
"""

from __future__ import annotations

__all__ = [
    "BoxMaskExtractor",
]

import argparse
import logging

import box
import cv2

from mon.core import (
    bbox as B,
    BBoxFormat,
    console,
    create_device,
    create_progress_bar,
    EXT,
    image as I,
    Path,
)
from mon.cv import segment

current_file = Path(__file__).normalize(exist=True)
current_dir  = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

class BoxMaskExtractor:
    """A runner for extracting box masks from images."""

    # --- Lifecycle & Initialization ---
    def __init__(self, cfg: box.Box):
        self._cfg      = cfg
        self.verbose   = cfg.verbose
        self._data_dir = Path(cfg.data_dir).normalize(exist=True)
        self._device   = create_device(cfg.device)
        self._model    = segment.SAMBoxSegmentor(
            name    = cfg.model,
            device  = self._device,
            verbose = self.verbose
        )

    # --- Callable & Context Manager ---
    def run(self):
        """Run the metric measurement process."""
        # Summarize the current run
        if not self.verbose:
            logger = logging.getLogger()
            logger.disabled = True
        console.rule(f"[bold red] {self._model}")
        console.log(f"[bold green]Model : {self._model}")
        console.log(f"[bold red]Data  : {str(self._data_dir)}")
        console.log(f"[bold]Device: {self._device}")

        # Recursively get all 'image' directories in the data directory
        image_dirs = self._data_dir.image_dirs(recursive=True)

        # Processing loop
        for image_dir in image_dirs:
            # Process each image directory
            label_dir  = image_dir.parent / "label"
            output_dir = image_dir.parent / "mask"

            # Skip if the label directory does not exist
            if label_dir.exists():
                self._process_dir(
                    image_dir  = image_dir,
                    label_dir  = label_dir,
                    output_dir = output_dir,
                )

    def _process_dir(
        self,
        image_dir : str | Path,
        label_dir : str | Path,
        output_dir: str | Path,
    ):
        """Process a single image directory.

        Args:
            image_dir: Path to the directory containing images.
            label_dir: Path to the directory containing label files.
            output_dir: Path to the directory where masks will be saved.
        """
        # Resolve paths
        image_dir = Path(image_dir).normalize()
        label_dir = Path(label_dir).normalize()

        if not image_dir.exists() or not label_dir.exists():
            return

        output_dir = Path(output_dir).normalize(mkdir=True)

        # List all image files
        image_files = sorted([f for f in list(image_dir.rglob("*")) if f.is_image_file(exist=True)])

        # Processing loop
        with create_progress_bar() as pbar:
            for i, image_file in pbar.track(
                sequence    = enumerate(image_files),
                total       = len(image_files),
                description = "[bright_yellow]Processing"
            ):
                # Read image
                image = cv2.imread(str(image_file))
                imgsz = I.imgsz(image)

                # Resolve label file
                label_file = label_dir / f"{image_file.stem}.txt"
                label_file = label_file.normalize(exist=True)

                # Read bounding boxes
                bbox = B.load(path=label_file, imgsz=imgsz, fmt=BBoxFormat.YOLO, verbose=False)

                # Extract masks for each bounding box
                masks = self._model.process(image=image, bbox=bbox)
                assert len(masks) == len(bbox), f"Expected {len(bbox)} masks, but got {len(masks)}."

                # Save masks
                mask_dir = (output_dir / image_file.stem).normalize(mkdir=True)
                for j, mask in enumerate(masks):
                    mask_file = mask_dir / f"{j}.{EXT.IMAGE}"
                    cv2.imwrite(str(mask_file), mask)

    # --- CLI ---
    @staticmethod
    def parse_args() -> box.Box:
        """Parse command line arguments.

        Returns:
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(description="extract_box_mask")
        parser.add_argument("--data_dir", type=str, required=True, help="The directory contains 'image', 'label', and 'mask' subdirectories.")
        parser.add_argument("--model",    type=str, default="sam2.1_b")
        parser.add_argument("--device",   type=str, default="cuda:0")
        parser.add_argument("--verbose",  action="store_true")
        return box.Box(vars(parser.parse_args()))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    args   = BoxMaskExtractor.parse_args()
    runner = BoxMaskExtractor(args)
    runner.run()

# endregion
