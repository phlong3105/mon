#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module runs the dehazing procedure."""

from __future__ import annotations

import argparse

import munch

import mon
import zid


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/train/detection/haze/images",
        help    = "Input source."
    )
    parser.add_argument(
        "--dst",
        type    = str,
        default = None,
        help    = "Output destination."
    )
    parser.add_argument(
        "--verbose",
        action = "store_true",
        help   = "Display results."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = munch.Munch.fromDict(vars(parse_args()))
    assert args.src is not None and mon.Path(args.src).is_dir()
    src  = mon.Path(args.src)
    if args.dst is None:
        dst = src.parent / "dehaze"
    else:
        dst = mon.Path(args.dst)
    assert mon.Path(dst).is_dir()
    
    image_files = list(src.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    
    with mon.progress_bar() as pbar:
        for f in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Converting"
        ):
            image = mon.read_image_cv(
                path      = str(f),
                to_rgb    = True,
                to_tensor = False,
                normalize = False,
            )
            name = f.name
            
            # Run
            zid.dehaze(image_name=name, image=image, num_iter=500, output_path=dst)
            
# endregion
