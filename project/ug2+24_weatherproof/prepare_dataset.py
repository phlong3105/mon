#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prepare and organize the raw dataset into my format."""

from __future__ import annotations

import shutil

import click
import cv2

import mon


# region Function

@click.command()
@click.option("--dataset-dir", type=click.Path(exists=True), default=mon.DATA_DIR/"ug2+/ug2+24_weatherproof/test", help="Dataset directory.")
@click.option("--verbose",     is_flag=True)
def main(
    dataset_dir: mon.Path,
    verbose    : bool
):
    assert dataset_dir is not None and mon.Path(dataset_dir).is_dir()
    dataset_dir = mon.Path(dataset_dir)
    subdirs     = sorted(list(dataset_dir.subdirs()))
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(subdirs)),
            total       = len(subdirs),
            description = f"[bright_yellow] Preparing"
        ):
            subdir     = subdirs[i]
            lq_dir     = subdir / "lq"
            hq_dir     = subdir / "hq"
            intern_dir = subdir / "intern"
            lq_dir.mkdir(parents=True, exist_ok=True)
            hq_dir.mkdir(parents=True, exist_ok=True)
            intern_dir.mkdir(parents=True, exist_ok=True)
            
            image_files = list(subdir.rglob("*"))
            image_files = [f for f in image_files if f.is_image_file()]
            image_files = sorted(image_files)
            for image_file in image_files:
                if "degraded" in str(image_file):
                    destination = lq_dir / image_file.name
                elif "clean" in str(image_file):
                    destination = hq_dir / image_file.name
                else:
                    destination = intern_dir / image_file.name
                shutil.move(image_file, destination)
                
# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
