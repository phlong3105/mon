#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/vis-opt-group/SCI

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.utils
import torchvision
from PIL import Image
from torch.autograd import Variable

import mon
from model import Finetunemodel

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, "png")


def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Model
    model = Finetunemodel(weights)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
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
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint.get("image")
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                input      = Variable(image).to(device)
                
                # Infer
                timer.tick()
                i, r = model(input)
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
                    torchvision.utils.save_image(r, str(output_path))
       
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    args.weights = args.weights or mon.ZOO_DIR / "vision/enhance/llie/sci/sci/lol_v1/sci_lol_v1_pretrained.pt"
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
