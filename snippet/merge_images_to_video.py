#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from munch import Munch

from one.core import InterpolationMode
from one.core import is_image_file
from one.core import is_video_file
from one.core import progress_bar
from one.core import to_size


# H1: - Functional -------------------------------------------------------------

def merge_images(args: Munch | dict):
    """
    Merge images into video.
    """
    source      = Path(args.source)
    image_files = list(source.rglob("*"))
    image_files = [f for f in image_files if is_image_file(f)]
    images      = []
    with progress_bar() as pbar:
        for f in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Listing"
        ):
            image   = cv2.imread(f)
            h, w, c = image.shape
            size    = (h, w)
            images.append(image)
    
    size   = to_size(size=args.size or size)  # [H, W]
    size   = size[::-1]                       # [W, H]
    dest   = str(args.dest) or f"{str(source.name)}.mp4"
    fps    = float(args.fps)
    writer = cv2.VideoWriter(
        filename  = dest,
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v"),
        fps       = fps,
        frameSize = size,
        isColor   = True,
    )
    interpolation = InterpolationMode.from_value(args.interpolation or "nearest")
    interpolation = InterpolationMode.cv_modes_mapping()[interpolation]
    
    with progress_bar() as pbar:
        for img in pbar.track(
            sequence    = images,
            total       = len(images),
            description = f"[bright_yellow] Processing"
        ):
            img_size  =to_size(img.shape)
            img_size = img_size[::-1]
            if img_size != size:
                img = cv2.resize(
                    src           = img,
                    dsize         = size,
                    fx            = 0,
                    fy            = 0,
                    interpolation = interpolation,
                )
                writer.write(img)
                if args.verbose:
                    cv2.imshow("Video", img)
            else:
                break
   
    writer.release()
    cv2.destroyAllWindows()


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        type=str,            default="",        help="Input images source (folder or pattern).")
    parser.add_argument("--dest",          type=str,            default=None,      help="Output video destination.")
    parser.add_argument("--size",          type=int, nargs="+", default=None,      help="Video sizes in [H, W] format.")
    parser.add_argument("--fps",           type=float,          default=20,        help="Output video frame rate.")
    parser.add_argument("--interpolation", type=str,            default="nearest", help="Interpolation method. One of: [nearest, linear, area, cubic, lanczos4].")
    parser.add_argument("--verbose",       action="store_true",                    help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    input_args    = vars(parse_args())
    source        = input_args.get("source",        None)
    dest          = input_args.get("dest",          None)
    size          = input_args.get("size",          None)
    fps           = input_args.get("fps",           None)
    interpolation = input_args.get("interpolation", None)
    verbose       = input_args.get("verbose",       False)
    args = Munch(
        source        = source,
        dest          = dest,
        size          = size,
        fps           = fps,
        interpolation = interpolation,
        verbose       = verbose,
    )
    merge_images(args=args)
