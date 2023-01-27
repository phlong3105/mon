#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module runs the dehazing procedure."""

from __future__ import annotations

import argparse

import munch

import mon
import zid
import gunet


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type    = str,
        default = mon.DATA_DIR / "a2i2-haze/train/detection/haze/images",
        help    = "Image directory."
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = None,
        help    = "Output directory."
    )
    parser.add_argument(
        "--model",
        type    = str,
        default = "gunet_b",
        help    = "Model name."
    )
    parser.add_argument(
        "--weights",
        type    = str,
        default = mon.Path(__file__).resolve().parent / "gunet/weights/haze4k/gunet_b.pth",
        help    = "Weights path."
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
    # Run
    # zid.dehaze(args=args, num_iter=500)
    gunet.detect(args=args)
    
# endregion
