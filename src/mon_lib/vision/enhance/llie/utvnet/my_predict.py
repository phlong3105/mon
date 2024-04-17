#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/CharlieZCJ/UTVNet

'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}

Tested with Pytorch 1.7.1, Python 3.6

Author: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''

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

import mon
from models import network

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

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
    device    = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device    = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark
    if benchmark:
        model = network.UTVNet().to(device)
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
    model = network.UTVNet().to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval()
    
    # Predicting
    torch.set_grad_enabled(False)
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path     = meta["path"]
                image          = Image.open(image_path).convert("RGB")
                image          = (np.asarray(image) / 255.0)
                image          = torch.from_numpy(image).float()
                image          = image.permute(2, 0, 1)
                image          = image.to(device).unsqueeze(0)
                start_time     = time.time()
                enhanced_image = model(image)
                enhanced_image = enhanced_image.clamp(0, 1).cpu()
                run_time       = (time.time() - start_time)
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
        
    """
    if args.input_dir_name == 'sRGBSID':
        test_input_dir = './dataset/sRGBSID/test/1/'
        test_input_dir2 = './dataset/sRGBSID/test/2/'
        test_gt_dir = './dataset/sRGBSID/gt/test/'
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '512', args.input_dir_name)

    else:
        test_input_dir = './dataset/ELD/{}/'.format(args.input_dir_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/ELD/{}g/'.format(args.input_dir_name)
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '1024', args.input_dir_name)

    test(model, args, loaderTest, device)
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
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = weights or mon.ZOO_DIR / "vision/enhance/llie/utvnet/utvnet_model_test.pt"
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
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
