#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch
import torch.optim

import mon
from net.model import AirNet
from utils.image_io import save_image_tensor

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data       = args.data
    save_dir   = args.save_dir
    weights    = args.weights
    device     = mon.set_device(args.device)
    epochs     = args.epochs
    imgsz      = args.imgsz[0]
    resize     = args.resize
    benchmark  = args.benchmark
    mode       = args.mode
    batch_size = args.batch_size
    opt        = argparse.Namespace(
        **{
            "mode"      : mode,
            "batch_size": batch_size,
        }
    )
    
    # Model
    model = AirNet(opt)
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = model,
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
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
            image_path  = meta["path"]
            timer.tick()
            restored    = model(x_query=image, x_key=image)
            timer.tock()
            output_path = save_dir / image_path.name
            save_image_tensor(restored, output_path)
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
