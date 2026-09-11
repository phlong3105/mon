#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Merge Images to Video.

This script provides a CLI for merging all images in a given directory into a single video.
The images are expected to be in the same directory and will be processed in alphabetical order.

Example:
    python merge_images_to_video.py --data "/Users/longpham/Downloads/vis_relabel" --video "output_video.mp4" --fps 24
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

import cv2

from mon import create_progress_bar, Path

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def merge_images_to_video(args: argparse.Namespace):
    """"""
    # Validate inputs
    data_dir = Path(args.data).normalize()
    output_video = Path(args.video).normalize()
    fps = args.fps

    if not data_dir.exists():
        print(f" Error: Image folder does not exist: {data_dir}")
        return
    if not data_dir.is_dir():
        print(f"Error: Image folder is not a directory: {data_dir}")
        return
    if fps <= 0:
        print(f"Error: FPS must be a positive integer: {args.fps}")
        return

    # Get and sort images sequentially
    image_files = list(data_dir.glob("*"))
    image_files = [i for i in image_files if i.is_image_file(exists=True)]
    image_files.sort()

    if args.imgsz:
        width, height = map(int, args.imgsz.split(","))
    else:
        # Read the first image to determine video dimensions
        first_image_path = image_files[0]
        frame = cv2.imread(str(first_image_path))
        height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    # 'mp4v' is widely compatible with .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, args.fps, (width, height))

    # Loop through images and add to video
    print(f"Starting video creation: {args.video} ({width}x{height}, {args.fps} FPS)...")
    with create_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(image_files)),
            total       = len(image_files),
            description = f"[bright_yellow]Merging images"
        ):
            image_file = image_files[i]
            frame = cv2.imread(str(image_file))
            if frame is None:
                print(f"Error reading image: {image_file}")
                continue
            # Optional: Resize image if it doesn't match the first image's dimensions
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            video.write(frame)

    # Release the video writer resource
    video.release()
    print("Video saved successfully!")

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    merge_images_to_video(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--data",  type=str, default="",                 help="Directory containing images to merge into video.")
    parser.add_argument("--video", type=str, default="output_video.mp4", help="Output video file name.")
    parser.add_argument("--fps",   type=int, default=24,                 help="Frames per second for the video.")
    parser.add_argument("--imgsz", type=str, default="1280,720",         help="Image size (width, height) for the video.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
