#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/xiwang-online/LLUnetPlusPlus>`__
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import mon
from model import NestedUNet

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    imgsz     = mon.parse_hw(imgsz)
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    model = NestedUNet().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(model),
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
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                meta       = datapoint.get("meta")
                image_path = meta["path"]
                image      = Image.open(image_path).convert("RGB")
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float()
                image      = image.permute(2, 0, 1)
                image      = image.to(device).unsqueeze(0)
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize_divisible(image, 32)
                timer.tick()
                enhanced_image = model(image)
                timer.tock()
                enhanced_image = mon.resize(enhanced_image, (h0, w0))
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
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
