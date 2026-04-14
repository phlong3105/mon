#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metric Evaluation Script.

This script provides a simple interface for measuring IQA metrics.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import Path
from mon.runners import InstanceIQAEvaluator

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    # 1. Define arguments
    metrics = ["psnr", "ssim", "ssimc", "lpips", "niqe", "pi", "ciqs", "giqs"]

    # 2. Measure metrics
    iqa = InstanceIQAEvaluator.from_cli(
        input_dir=args.input_dir,
        result_file=None,
        metrics=metrics,
        device="cuda:0",
        imgsz=args.imgsz,
        resize=args.resize,
        verbose=True,
    )
    iqa.measure()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--input-dir", type=str, default="/home/longpham/_/code/mon/projects/dev/run/assets/sice_112/")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--resize", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
