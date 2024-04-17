#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/vis-opt-group/SCI

from __future__ import annotations

import argparse
import copy
import os
import socket
import sys
import time

import click
import numpy as np
import torch
import torch.utils
import torchvision
from PIL import Image
from torch.autograd import Variable

import mon
from model import Finetunemodel

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, "png")


def predict(args: argparse.Namespace):
    # General config
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Device
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = Finetunemodel(weights)
    model = model.to(device)
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
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path  = meta["path"]
                input       = Variable(images).to(device)
                start_time  = time.time()
                i, r        = model(input)
                run_time    = time.time() - start_time
                sum_time   += run_time
                output_path = save_dir / image_path.name
                torchvision.utils.save_image(r, str(output_path))
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = weights or mon.ZOO_DIR / "vision/enhance/llie/sci_medium.pt"
    weights  = mon.to_list(weights)
    project  = root.name
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args = {
        "root"      : root,
        "config"    : config,
        "weights"   : weights,
        "model"     : model,
        "data"      : data,
        "project"   : project,
        "name"      : fullname,
        "save_dir"  : save_dir,
        "device"    : device,
        "imgsz"     : imgsz,
        "resize"    : resize,
        "benchmark" : benchmark,
        "save_image": save_image,
        "verbose"   : verbose,
        "seed"      : 2,
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
