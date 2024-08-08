#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/Ian0926/DCC-Net

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import mon
import src.model as mmodel

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
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    color_net = mmodel.color_net().to(device)
    # color_net = mon.DataParallel(color_net)
    color_net.load_state_dict(torch.load(weights))
    color_net.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(color_net),
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
            for image, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path     = meta["path"]
                data_lowlight  = Image.open(image_path)
                data_lowlight  = (np.asarray(data_lowlight) / 255.0)
                data_lowlight  = torch.from_numpy(data_lowlight).float()
                data_lowlight  = data_lowlight.permute(2, 0, 1)
                data_lowlight  = data_lowlight.cuda().unsqueeze(0)
                h, w           = mon.get_image_size(data_lowlight)
                data_lowlight  = mon.resize_divisible(data_lowlight, 32)
                timer.tick()
                gray, color_hist, enhanced_image = color_net(data_lowlight)
                timer.tock()
                enhanced_image = mon.resize(enhanced_image, (h, w))
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
