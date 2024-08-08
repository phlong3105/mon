#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/KarelZhang/RUAS

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import mon
from model import Network

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    device    = mon.set_device(args.device)
    seed      = args.seed
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Seed
    mon.set_random_seed(seed)
    
    # Device
    cudnn.benchmark = True
    cudnn.enabled   = True
    
    # Model
    model = Network().to(device)
    model.load_state_dict(torch.load(str(weights)))
    for p in model.parameters():
        p.requires_grad = False
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
        to_tensor   = True,
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
                image_path = meta["path"]
                input      = image.to(device)
                timer.tick()
                u_list, r_list = model(input)
                timer.tock()
                output_path = save_dir / image_path.name
                save_images(u_list[-1], str(output_path))
                # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                """
                if args.model == "lol":
                    save_images(u_list[-1], u_path)
                elif args.model == "upe" or args.model == "dark":
                    save_images(u_list[-2], u_path)
                """
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    args.weights = args.weights or mon.ZOO_DIR / "vision/enhance/llie/ruas/ruas/lol_v1/ruas_lol_v1_pretrained.pt"
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
