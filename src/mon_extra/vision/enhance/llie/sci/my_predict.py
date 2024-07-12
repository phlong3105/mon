#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/vis-opt-group/SCI

from __future__ import annotations

import argparse
import copy
import time

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
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
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

if __name__ == "__main__":
    args = mon.parse_predict_args()
    args.weights = args.weights or mon.ZOO_DIR / "vision/enhance/llie/sci/sci/lol_v1/sci_lol_v1_pretrained.pt"
    predict(args)

# endregion
