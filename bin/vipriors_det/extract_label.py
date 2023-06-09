#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract label."""

from __future__ import annotations

import click

import mon
from mon.foundation.file import json


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"vipriors/delftbikes/train/images", type=click.Path(exists=True),  help="Image directory.")
@click.option("--label_file", default=mon.DATA_DIR/"vipriors/delftbikes/train/trainval_annotations.json", type=click.Path(exists=True), help="JSON label file.")
@click.option("--output-dir", default=mon.DATA_DIR/"vipriors/delftbikes/train/labels", type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",    is_flag=True)
def extract_label(
    image_dir : mon.Path,
    label_file: mon.Path,
    output_dir: mon.Path,
    verbose   : bool
):
    assert image_dir  is not None and mon.Path(image_dir).is_dir()
    assert label_file is not None and mon.Path(label_file).is_json_file()

    image_dir   = mon.Path(image_dir)
    output_dir  = output_dir or image_dir.parent / f"labels"
    output_dir  = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)

    label_file  = mon.Path(label_file)
    with open(label_file, "r") as file:
        json_data = json.load(file)

    with mon.get_progress_bar() as pbar:
        for image in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Extracting"
        ):
            data        = json_data[image.name]
            output_file = output_dir / f"{image.stem}.txt"

            with open(output_file, "w") as out:
                for ind, i in enumerate(data["parts"], 0):
                    label = data["parts"][i]
                    # if label["object_state"] != "absent":
                    part_name          = label["part_name"]
                    loc                = label["absolute_bounding_box"]
                    xmin               = loc["left"]
                    xmax               = xmin + loc["width"]
                    ymin               = loc["top"]
                    ymax               = ymin + loc["height"]
                    trust              = label["trust"]
                    object_state       = label["object_state"]
                    object_state_class = label["object_state_class"]
                    out.write(f"{part_name} {xmin} {ymin} {xmax} {ymax} {trust} {object_state} {object_state_class}\n")

# endregion


# region Main

if __name__ == "__main__":
    extract_label()

# endregion
