#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detection scripts.
"""

from __future__ import annotations

import argparse
import os

from chalearn import pretrained_dir
from chalearn.ltd import run_detect


# MARK: - Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-cfg",      default="",                                  type=str, help="Predefined configs.")
    parser.add_argument("--months",       default="*",                                 nargs="+", type=str, help="Subset to run inference.")
    parser.add_argument("--run-async",    default=False,                               action="store_true", help="Detect in asynchronous mode.")
    parser.add_argument("--weights",      default=os.path.join(pretrained_dir, "scaled_yolov4", "yolov4_p7_coco.pt"), nargs="+", type=str, help="model.pt path(s)")
    parser.add_argument("--source",       default=os.path.join("inference", "output"), type=str,   help="source")  # file/folder, 0 for webcam
    parser.add_argument("--output",       default=os.path.join("inference", "output"), type=str,   help="output folder")  # output folder
    parser.add_argument("--img-size",     default=1536,                                type=int,   help="inference size (pixels)")
    parser.add_argument("--conf-thres",   default=0.4,                                 type=float, help="object confidence threshold")
    parser.add_argument("--iou-thres",    default=0.5,                                 type=float, help="IOU threshold for NMS")
    parser.add_argument("--device",       default="",                                  help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img",     default=True,                                action="store_true", help="display results")
    parser.add_argument("--save-txt",     default=False,                               action="store_true", help="save results to *.txt")
    parser.add_argument("--classes",                                                   nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", default=False,                               action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment",      default=False,                               action="store_true", help="augmented inference")
    parser.add_argument("--update",       default=False,                               action="store_true", help="update all models")
    parser.add_argument("--verbose",      default=True,                                action="store_true", help="")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    run_detect(args)
    # merge_pkls()
