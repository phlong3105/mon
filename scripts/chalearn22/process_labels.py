#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Labels processing scripts.
"""

from __future__ import annotations

import argparse

from one.data.datasets.chalearn import convert_yolo_labels
from one.data.datasets.chalearn import convert_yolo_labels_asynchronous
from one.data.datasets.chalearn import shuffle_images_labels


# MARK: - Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",      default="train", type=str)
    parser.add_argument("--convert",    default=False,   action="store_true", help="Convert labels to YOLO format.")
    parser.add_argument("--save_image", default=False,   action="store_true", help="Should save image with drawn bounding boxes.")
    parser.add_argument("--verbose",    default=False,   action="store_true", help="Should show image with drawn bounding boxes.")
    parser.add_argument("--asynch",     default=True,    action="store_true", help="Run labels conversion in asynchronous mode.")
    parser.add_argument("--shuffle",    default=True,    action="store_true", help="Shuffle and copy images and labels to train/val folders.")
    parser.add_argument("--subset",     default="month", type=str,            help="Subset to shuffle and copy.")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.convert:
        if args.asynch:
            convert_yolo_labels_asynchronous(**vars(args))
        else:
            convert_yolo_labels(**vars(args))
    if args.shuffle:
        shuffle_images_labels(**vars(args))
