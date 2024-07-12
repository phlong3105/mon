#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket
import time

import click

import mon
from model import RetinexNet

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    # weights   = args.weights
    # weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    weights   = mon.ZOO_DIR / "vision/enhance/llie/retinexnet/retinexnet/lol_v1"
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    model = RetinexNet(imgsz, benchmark).to(device)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    with mon.get_progress_bar() as pbar:
        for images, target, meta in pbar.track(
            sequence    = data_loader,
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            image_path  = meta["path"]
            image_paths.append(image_path)
    
    # Predicting
    start_time = time.time()
    model.predict(
        image_paths,
        res_dir  = str(save_dir),
        ckpt_dir = str(weights),
    )
    run_time   = (time.time() - start_time)
    avg_time   = float(run_time / len(image_paths))
    console.log(f"Average time: {avg_time}")

# endregion


# region Main

if __name__ == "__main__":
    args = mon.parse_predict_args()
    predict(args)

# endregion
