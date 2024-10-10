#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script process SICE datasets."""

from __future__ import annotations

import cv2

import mon


def arrange_image(split: str = "test", level: str = "1"):
    input_dir   = mon.DATA_DIR / f"enhance/llie/sice/{split}/raw"
    output_dir  = mon.DATA_DIR / f"enhance/llie/sice/{split}/image"
    image_files = list(input_dir.rglob(f"{level}.jpg"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            path           = image_files[i]
            exposure_level = path.stem
            image_id       = path.parent.stem
            output_path    = output_dir / f"{image_id}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image = cv2.imread(str(image_files[i]))
            cv2.imwrite(str(output_path), image)


arrange_image("train", "1")
