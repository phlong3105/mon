#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from munch import Munch

from one.core import progress_bar

CURRENT_DIR = Path(__file__).parent.absolute()


# H1: - Functional -------------------------------------------------------------

def generate_submission(args: dict | Munch | argparse.Namespace):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    source      = Path(args.source)
    output_file = Path(args.output_file)
    output_data = []
    
    file_list   = list(sorted(os.listdir(os.path.join(source, "labels"))))
    file_list   = [source / "labels" / f for f in file_list]
    # file_list   = list(source.rglob("*.txt"))
    # file_list   = sorted(file_list)
    
    with progress_bar() as pbar:
        for i in pbar.track(
            range(len(file_list)),
            description=f"[bright_yellow]Processing files"
        ):
            path     = file_list[i]
            image_id = str(path.name).replace(".txt", ".jpg")
            lines    = open(path, "r").read().splitlines()
            for l in lines:
                d = l.split(" ")
                output_data.append(
                    {
                        "image_id"   : i,
                        "category_id": int(d[0]) + 1,
                        "bbox"       : [float(d[1]), float(d[2]), float(d[3]) - float(d[1]), float(d[4]) - float(d[2])],
                        "score"      : float(d[5])
                    }
                )
    
    with open(output_file, "w") as f:
        json.dump(output_data, f)
    

# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",      default=CURRENT_DIR/"runs"/"detect"/"yolov7-e6e-delftbikes-1280", type=str, help="Directory containing YOLO results .txt")
    parser.add_argument("--output-file", default=CURRENT_DIR/"runs"/"detect"/"submission.json",            type=str, help="Submission .json file")
    args   = parser.parse_args()
    return args


if __name__ == "__main__":
    generate_submission(parse_args())
