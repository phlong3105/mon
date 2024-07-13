#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/DavidQiuChao/PIE

from __future__ import annotations

import argparse
import socket
import time

import click
import cv2

import mon
import pie

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    sum_time = 0
    with mon.get_progress_bar() as pbar:
        for images, target, meta in pbar.track(
            sequence    = data_loader,
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            image_path     = meta["path"]
            image          = cv2.imread(str(image_path))
            start_time     = time.time()
            enhanced_image = pie.PIE(image)
            run_time       = (time.time() - start_time)
            output_path    = save_dir / image_path.name
            cv2.imwrite(str(output_path), enhanced_image)
            sum_time      += run_time
    avg_time = float(sum_time / len(data_loader))
    console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
