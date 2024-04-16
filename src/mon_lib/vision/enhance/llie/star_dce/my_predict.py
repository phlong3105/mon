#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/zzyfd/STAR-pytorch/tree/main>`__
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import time

import click
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision

import models.model
import mon

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# region Predict

def RGBtoYUV444(rgb):
    # code from Jun
    # yuv range: y[0,1], uv[-0.5, 0.5]
    height, width, ch = rgb.shape
    assert ch == 3, "rgb should have 3 channels"
    
    rgb2yuv_mat = np.array([[0.299, 0.587, 0.114], [-0.16874, -0.33126, 0.5], [0.5, -0.41869, -0.08131]], dtype=np.float32)
    rgb_t       = rgb.transpose(2, 0, 1).reshape(3, -1)
    yuv         = rgb2yuv_mat @ rgb_t
    yuv         = yuv.transpose().reshape((height, width, 3))
    
    # return yuv.astype(np.float32)
    # rescale uv to [0,1]
    yuv[:, :, 1] += 0.5
    yuv[:, :, 2] += 0.5
    return yuv


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def predict(args: argparse.Namespace):
    # General config
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    device    = args.device
    imgsz     = args.imgsz
    imgsz     = mon.parse_hw(imgsz)
    resize    = args.resize
    benchmark = args.benchmark
    
    # Seed
    cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark
    if benchmark:
        model = models.model.enhance_net_litr()
        flops, params, avg_time = mon.calculate_efficiency_score(
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
        
    # Model
    DCE_net = models.model.enhance_net_litr().to(device)
    DCE_net.load_state_dict(torch.load(weights), strict=True)
    DCE_net.eval()
    
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
                image      = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image      = cv2.transpose(image)
                image      = (np.asarray(image[..., ::-1]) / 255.0)
                image      = RGBtoYUV444(image)
                image      = torch.from_numpy(image).float()  # float32
                image      = image.to(device).unsqueeze(0)
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(input=image, size=imgsz)
                else:
                    image = mon.resize_divisible(image=image, divisor=32)
                image_resize   = F.interpolate(image, (args.image_ds, args.image_ds), mode="area")
                start_time     = time.time()
                enhanced_image, x_r = DCE_net(image_resize, img_in=image)
                run_time       = (time.time() - start_time)
                enhanced_image = mon.resize(input=enhanced_image, size=[h0, w0])
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
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
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    weights  = weights  or args.get("weights")
    project  = args.get("project")
    fullname = fullname or args.get("name")
    device   = device   or args.get("device")
    imgsz    = imgsz    or args.get("imgsz")
    verbose  = verbose  or args.get("verbose")
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    project  = root.name or project
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["data"]       = data
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
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
