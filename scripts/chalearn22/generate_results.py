#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate final pkl results file.
"""

from __future__ import annotations

import argparse
import os

from chalearn.object_detection.scaled_yolov4 import create_pkl_from_txts
from chalearn.object_detection.scaled_yolov4 import merge_pkls


# MARK: - Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default=os.path.join("inference", "output"), type=str, help="Output path.")
    parser.add_argument("--from-txts",   default=True,                                action="store_true", help="Merge from YOLO's txt results.")
    parser.add_argument("--conf-thres",  default=0.5,                                 nargs="+", type=float, help="Confidence threshold. Can be a string, scalar value or a list/tuple/dict of values for each class.")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.from_txts:
        create_pkl_from_txts(
            output_path = args.output_path,
            conf_thres  = args.conf_thres
        )
    else:
        merge_pkls(output_path=args.output_path)
