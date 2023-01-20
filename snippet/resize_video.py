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
from one.core import is_video_file
from one.core import progress_bar
from one.core import to_size


# H1: - Functional -------------------------------------------------------------

def resize_video(args: Munch | dict):
    assert is_video_file(args.source)
    
    source        = Path(args.source)
    dest          = str(args.dest) or f"{str(source.name)}-resized{str(source.suffix)}"
    size          = to_size(size=args.size)  # [H, W]
    size          = size[::-1]               # [W, H]
    capture       = cv2.VideoCapture(str(source))
    num_frames    = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps           = float(args.fps) or float(capture.get(cv2.CAP_PROP_FPS))
    writer        = cv2.VideoWriter(
        filename  = dest,
        fourcc    = cv2.VideoWriter_fourcc(*"mp4v"),
        fps       = fps,
        frameSize = size,
        isColor   = True,
    )
    interpolation = InterpolationMode.from_value(args.interpolation or "nearest")
    interpolation = InterpolationMode.cv_modes_mapping()[interpolation]
    
    with progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(num_frames),
            total       = num_frames,
            description = f"[bright_yellow] Processing"
        ):
            ret, frame = capture.read()
            if ret is True:
                resized = cv2.resize(
                    src           = frame,
                    dsize         = size,
                    fx            = 0,
                    fy            = 0,
                    interpolation = interpolation,
                )
                writer.write(resized)
                if args.verbose:
                    cv2.imshow("Resize", resized)
            else:
                break
   
    capture.release()
    writer.release()
    cv2.destroyAllWindows()


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",        type=str,            default="",         help="Input video source.")
    parser.add_argument("--dest",          type=str,            default=None,       help="Output video destination.")
    parser.add_argument("--size",          type=int, nargs="+", default=[512, 512], help="Video sizes in [H, W] format.")
    parser.add_argument("--fps",           type=float,          default=None,       help="Output video frame rate.")
    parser.add_argument("--interpolation", type=str,            default="nearest",  help="Interpolation method. One of: [nearest, linear, area, cubic, lanczos4].")
    parser.add_argument("--verbose",       action="store_true",                     help="Display results.")
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
    resize_video(args=args)
