#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/DavidQiuChao/PIE

from __future__ import annotations

import argparse

import cv2

import mon
import pie

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            image       = datapoint.get("image")
            meta        = datapoint.get("meta")
            image_path  = mon.Path(meta["path"])
            timer.tick()
            enhanced_image = pie.PIE(image)
            timer.tock()
            
            # Save
            if save_image:
                if use_fullpath:
                    rel_path = image_path.relative_path(data_name)
                    save_dir = save_dir / rel_path.parent
                else:
                    save_dir = save_dir / data_name
                output_path  = save_dir / image_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), enhanced_image)
    
    avg_time = float(timer.avg_time)
    console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
