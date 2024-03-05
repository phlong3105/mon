#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/DavidQiuChao/PIE

from __future__ import annotations

import argparse
import time

import cv2

import mon
import pie
from mon import RUN_DIR

console = mon.console


def main(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    #
    image_paths = list(args.input_dir.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for _, image_path in pbar.track(
            sequence    = enumerate(image_paths),
            total       = len(image_paths),
            description = f"[bright_yellow] Inferring"
        ):
            # console.log(image_path)
            image          = cv2.imread(str(image_path))
            start_time     = time.time()
            enhanced_image = pie.PIE(image)
            run_time       = (time.time() - start_time)
            output_path    = args.output_dir / image_path.name
            cv2.imwrite(str(output_path), enhanced_image)
            sum_time      += run_time
    avg_time = float(sum_time / len(image_paths))
    console.log(f"Average time: {avg_time}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, default="data/test_data/")
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/zerodce")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
