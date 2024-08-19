#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import time
from copy import deepcopy

import cv2
import numpy as np
import thop
import torch
import torch.optim
import torchvision
from fvcore.nn import FlopCountAnalysis, parameter_count
from PIL import Image
from torch import nn

import mon
from model import Image_network
from mon import core
from mon.core import _size_2_t

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def get_hist(file_name):
    src    = cv2.imread(file_name)
    src    = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    hist_s = np.zeros((3, 256))

    for (j, color) in enumerate(("red", "green", "blue")):
        s = src[..., j]
        hist_s[j, ...], _ = np.histogram(s.flatten(), 256, [0, 256])
        hist_s[j, ...]    = hist_s[j, ...] / np.sum(hist_s[j, ...])

    hist_s = torch.from_numpy(hist_s).float()

    return hist_s


def calculate_efficiency_score(
    model     : nn.Module,
    image_size: _size_2_t = 512,
    channels  : int       = 3,
    runs      : int       = 100,
    use_cuda  : bool      = True,
    verbose   : bool      = False,
):
    # Define input tensor
    h, w  = core.parse_hw(image_size)
    input = torch.rand(1, channels, h, w)
    hist  = np.zeros((3, 256))
    hist  = torch.from_numpy(hist).float()
    hist  = hist.unsqueeze(0)
    
    # Deploy to cuda
    if use_cuda:
        input = input.cuda()
        hist  = hist.cuda()
        model = model.cuda()
    
    # Get FLOPs and Params
    flops, params = thop.profile(deepcopy(model), inputs=(input, hist), verbose=verbose)
    flops         = FlopCountAnalysis(model, (input, hist)).total() if flops == 0 else flops
    params        = model.params if hasattr(model, "params") and params == 0 else params
    params        = parameter_count(model) if hasattr(model, "params") else params
    params        = sum(list(params.values())) if isinstance(params, dict) else params
    g_flops       = flops * 1e-9
    m_params      = int(params) * 1e-6
    
    # Get time
    start_time = time.time()
    for i in range(runs):
        _ = model(input, hist)
    runtime    = time.time() - start_time
    avg_time   = runtime / runs
    
    # Print
    if verbose:
        console.log(f"FLOPs (G) : {flops:.4f}")
        console.log(f"Params (M): {params:.4f}")
        console.log(f"Time (s)  : {avg_time:.4f}")
    
    return flops, params, avg_time


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
    Imgnet = Image_network().to(device)
    Imgnet.load_state_dict(torch.load(weights))
    Imgnet.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = calculate_efficiency_score(
            model      = copy.deepcopy(Imgnet),
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
                image_path = mon.Path(meta["path"])
                image      = Image.open(str(image_path))
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float()
                image      = image.permute(2, 0, 1)
                image      = image.to(device).unsqueeze(0)
                histogram  = get_hist(str(image_path))
                histogram  = histogram.to(device).unsqueeze(0)
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize_divisible(image, 32)
                timer.tick()
                enhanced_image, vec, wm, xy = Imgnet(image, histogram)
                timer.tock()
                enhanced_image = mon.resize(enhanced_image, (h0, w0))
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path = image_path.relative_path(data_name)
                        save_dir = save_dir / rel_path.parent
                    else:
                        save_dir = save_dir / data_name
                    output_path  = save_dir / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
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
