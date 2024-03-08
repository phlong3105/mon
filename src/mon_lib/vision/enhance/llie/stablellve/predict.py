#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/zkawfanx/StableLLVE

from __future__ import annotations

import argparse
import os
import socket
import time

import click
import numpy as np
import torch
import torchvision
from PIL import Image

from model import UNet
from mon import core, data as d, nn
from mon.globals import ZOO_DIR

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

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
    
    # Load model
    model = UNet(n_channels=3, bilinear=True)
    model.load_state_dict(torch.load(weights))
    model = model.to(device)
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = nn.calculate_efficiency_score(
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
                image_path     = meta["image_path"]
                image          = Image.open(image_path).convert("RGB")
                image          = (np.asarray(image) / 255.0)
                image          = torch.from_numpy(image).float()
                image          = image.permute(2, 0, 1)
                image          = image.to(device).unsqueeze(0)
                start_time     = time.time()
                enhanced_image = model(image)
                run_time       = (time.time() - start_time)
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
        
    """
    with torch.no_grad():
        for i, filename in enumerate(filenames):
            test = cv2.imread(filename)/255.0
            test = np.expand_dims(test.transpose([2,0,1]), axis=0)
            test = torch.from_numpy(test).to(device="cuda", dtype=torch.float32)
            out  = model(test)
            out  = out.to(device="cpu").numpy().squeeze()
            out  = np.clip(out*255.0, 0, 255)
            path = filename.replace('/test/','/results/')[:-4]+'.png'
            # folder = os.path.dirname(path)
            # if not os.path.exists(folder):
            #     os.makedirs(folder)
            cv2.imwrite(path, out.astype(np.uint8).transpose([1,2,0]))
            print('%d|%d'%(i+1, len(filenames)))
    """

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
    
    # Prioritize input args --> config file args
    root     = core.Path(root)
    weights  = weights or ZOO_DIR / "vision/enhance/llie/stablellve/stablellve_checkpoint.pth"
    project  = root.name
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = core.parse_device(device)
    # imgsz    = core.str_to_int_list(imgsz)
    # imgsz    = [int(i) for i in imgsz]
    imgsz    = core.parse_hw(imgsz)[0]
    
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
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return save_dir
    
    
if __name__ == "__main__":
    main()

# endregion
