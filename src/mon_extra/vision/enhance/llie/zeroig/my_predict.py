#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import sys
import time

import numpy as np
import torch
import torch.optim
import torch.utils
from PIL import Image
from thop import profile
from torch.autograd import Variable
from torchvision import transforms

import mon
from model import Finetunemodel

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    flops, _           = profile(model, inputs=(input_tensor,))
    flops_in_gigaflops = flops / 1e9  # Convert FLOPs to gigaflops (G)
    return flops_in_gigaflops


def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    seed         = args.seed
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    use_fullpath = args.use_fullpath
    mon.set_random_seed(seed)
    
    # Model
    if not torch.cuda.is_available():
        console.log("No gpu device available")
        sys.exit(1)
    
    model = Finetunemodel(str(weights))
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
        total_params = calculate_model_parameters(model)
        console.log(f"FLOPs        = {flops:.4f}")
        console.log(f"Params       = {params:.4f}")
        console.log(f"Time         = {avg_time:.4f}")
        console.log(f"Total Params = {total_params:.4f}")
        
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
    for p in model.parameters():
        p.requires_grad = False
    
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                meta            = datapoint.get("meta")
                image_path      = mon.Path(meta["path"])
                data_lowlight   = Image.open(image_path).convert("RGB")
                # data_lowlight   = transforms.ToTensor()(data_lowlight).numpy()
                # data_lowlight   = np.transpose(data_lowlight, (1, 2, 0))
                # data_lowlight   = np.asarray(data_lowlight, dtype=np.float32)
                # data_lowlight   = np.transpose(data_lowlight[:, :, :], (2, 0, 1))
                # data_lowlight   = torch.from_numpy(data_lowlight)
                data_lowlight   = (np.asarray(data_lowlight) / 255.0)
                data_lowlight   = torch.from_numpy(data_lowlight).float()
                data_lowlight   = data_lowlight.permute(2, 0, 1)
                data_lowlight   = data_lowlight.to(device).unsqueeze(0)
                input           = Variable(data_lowlight, volatile=True).to(device)
                timer.tick()
                enhance, output = model(input)
                timer.tock()
                
                # Save
                if use_fullpath:
                    rel_path   = image_path.relative_path(data_name)
                    output_dir = save_dir / rel_path.parents[0]
                    debug_dir  = save_dir / rel_path.parents[1] / f"{rel_path.parent.name}_denoise"
                else:
                    output_dir = save_dir / data_name
                    debug_dir  = save_dir / f"{data_name}_denoise"
                output_path    = save_dir / image_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                enhance = save_images(enhance)
                output  = save_images(output)
                Image.fromarray(output).save(str(debug_dir   / f"{image_path.stem}.png"), "PNG")
                Image.fromarray(enhance).save(str(output_dir / f"{image_path.stem}.png"), "PNG")
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
