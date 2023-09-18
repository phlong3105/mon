#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/DavidQiuChao/PIE

from __future__ import annotations

import argparse
import time

import cv2

import mon
import pie

console = mon.console


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default="data/test_data/")
    parser.add_argument("--weights",    type=str, default="weights/Epoch99.pth")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=mon.RUN_DIR/"predict/zerodce")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    #
    image_paths = list(args.data.rglob("*"))
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
            result_path    = args.output_dir / image_path.name
            cv2.imwrite(str(result_path), enhanced_image)
            sum_time      += run_time
    avg_time = float(sum_time / len(image_paths))
    console.log(f"Average time: {avg_time}")


if __name__ == "__main__":
    main()
