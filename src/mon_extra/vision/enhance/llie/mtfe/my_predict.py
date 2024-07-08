#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import socket
import time
from copy import deepcopy

import click
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

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


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
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    devices   = mon.set_device(args.devices)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    Imgnet = Image_network().to(devices)
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
                image_path = meta["path"]
                image      = Image.open(str(image_path))
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float()
                image      = image.permute(2, 0, 1)
                image      = image.to(devices).unsqueeze(0)
                histogram  = get_hist(str(image_path))
                histogram  = histogram.to(devices).unsqueeze(0)
                
                h0, w0 = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize_divisible(image, 32)
              
                start_time = time.time()
                enhanced_image, vec, wm, xy = Imgnet(image, histogram)
                run_time   = (time.time() - start_time)
                
                enhanced_image = mon.resize(enhanced_image, (h0, w0))
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                
                sum_time     += run_time
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
@click.option("--devices",    type=str, default=None, help="Running devices.")
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
    devices   : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    weights  = weights  or args.get("weights")
    fullname = fullname or args.get("name")
    devices  = devices  or args.get("devices")
    imgsz    = imgsz    or args.get("imgsz")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    devices  = mon.parse_device(devices)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["devices"]    = devices
    args["imgsz"]      = imgsz
    args["resize"]     = resize
    args["benchmark"]  = benchmark
    args["save_image"] = save_image
    args["verbose"]    = verbose
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
