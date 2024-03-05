#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket
import time

import click
import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model as mmodel
from mon import core, data as d, nn

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def run_infer(weights, image_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net    = mmodel.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(weights))
    start_time = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    run_time   = (time.time() - start_time)
    # print(run_time)
    '''
    image_path  = image_path.replace("test_data", "result")
    result_path = image_path
    if not os.path.exists(image_path.replace("/" + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace("/" + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(enhanced_image, result_path)
    '''
    return enhanced_image, run_time


def predict(args: argparse.Namespace):
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = args.save_dir
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Benchmark
    if benchmark:
        DCE_net = mmodel.enhance_net_nopool().to(device)
        DCE_net.load_state_dict(torch.load(weights))
        flops, params, avg_time = nn.calculate_efficiency_score(
            model      = DCE_net,
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
    console.log(f"{data}")
    data_name, data_loader, data_writer = d.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    with torch.no_grad():
        sum_time = 0
        with core.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path = meta["image_path"]
                enhanced_image, run_time = run_infer(weights, image_path)
                output_path = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time += run_time
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
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = core.Path(root)
    weights  = weights   or args["weights"]
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = device   or args["device"]
    device   = core.parse_device(device)
    imgsz    = imgsz    or args["imgsz"]
    # imgsz    = core.str_to_int_list(imgsz)
    # imgsz    = [int(i) for i in imgsz]
    imgsz    = core.parse_hw(imgsz)[0]
    verbose  = verbose  or args["verbose"]
    
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