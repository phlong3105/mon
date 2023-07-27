#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate COCO-format submission file."""

from __future__ import annotations

import click
import cv2

import mon
from mon.core.file import json


# region Function

@click.command()
@click.option("--image-dir",   default=mon.DATA_DIR/"vipriors/delftbikes/test/images", type=click.Path(exists=True),  help="Image directory.")
@click.option("--label-dir",   default=mon.RUN_DIR/"predict/delftbikes/submission/labels", type=click.Path(exists=True), help="Inference label directory.")
@click.option("--output-file", default=mon.RUN_DIR/"predict/delftbikes/submission/submission.json", type=click.Path(exists=False), help="Output JSON file.")
@click.option("--verbose",     is_flag=True)
def generate_submission(
    image_dir  : mon.Path,
    label_dir  : mon.Path,
    output_file: mon.Path,
    verbose    : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    output_file = output_file or label_dir.parent / f"submission.json"
    output_file = mon.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    
    label_files = list(label_dir.rglob("*"))
    label_files = [f for f in label_files if f.is_txt_file()]
    label_files = sorted(label_files)
    
    output_list = []
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(label_files)),
            total       = len(label_files),
            description = f"[bright_yellow] Processing"
        ):
            input_file = label_files[i]
            image_file = image_dir / f"{input_file.stem}.jpg"
            assert image_file.is_image_file()
            image   = cv2.imread(str(image_file))
            h, w, c = image.shape
            
            with open(input_file, "r") as input_f:
                lines   = input_f.read().splitlines()
                lines   = lines[::-1]  # Sort detection based on confidence score
                lines   = [x.strip().split(" ") for x in lines]
                # outputs = {}
                for l in lines:
                    l[3] = (float(l[3]) * w)
                    l[4] = (float(l[4]) * h)
                    l[1] = (float(l[1]) * w) - (l[3] / 2)
                    l[2] = (float(l[2]) * h) - (l[4] / 2)
                    image_id    = i
                    category_id = int(l[0]) + 1
                    bbox        = [l[1], l[2], l[3], l[4]]
                    score       = float(l[5])
                    '''
                    if category_id in outputs and score < outputs[category_id]["score"]:
                        continue
                    outputs[category_id] = {
                        "image_id"   : image_id,
                        "category_id": category_id,
                        "bbox"       : bbox,
                        "score"      : score,
                    }
                    '''
                    output_list.append(
                        {
                            "image_id"   : image_id,
                            "category_id": category_id,
                            "bbox"       : bbox,
                            "score"      : score,
                        }
                    )
                # output_list.extend(outputs.values())
                
    with open(output_file, "w") as output_f:
        json.dump(output_list, output_f)
    
        
# endregion


# region Main

if __name__ == "__main__":
    generate_submission()

# endregion
