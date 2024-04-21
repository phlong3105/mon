#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/pvnieo/Low-light-Image-Enhancement

from __future__ import annotations

import argparse
import socket
import time

import click
import cv2

import mon
from exposure_enhancement import enhance_image_exposure

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    weights   = args.weights[0] if isinstance(args.weights, list) else args.weights
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    sum_time = 0
    with mon.get_progress_bar() as pbar:
        for images, target, meta in pbar.track(
            sequence    = data_loader,
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            image_path = meta["path"]
            image      = cv2.imread(str(image_path))
            h, w, c    = image.shape
            if resize:
                image = cv2.resize(image, (imgsz, imgsz))
            start_time     = time.time()
            enhanced_image = enhance_image_exposure(
                im      = image,
                gamma   = args.gamma,
                lambda_ = args.lambda_,
                dual    = not args.lime,
                sigma   = args.sigma,
                bc      = args.bc,
                bs      = args.bs,
                be      = args.be,
                eps     = args.eps
            )
            run_time = (time.time() - start_time)
            if resize:
                enhanced_image = cv2.resize(enhanced_image, (w, h))
            output_path = save_dir / image_path.name
            cv2.imwrite(str(output_path), enhanced_image)
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
@click.option("--gamma",      type=float,   default=0.6,  help="Gamma correction parameter.")
@click.option("--lambda_",    type=float,   default=0.15, help="The weight for balancing the two terms in the illumination refinement optimization objective.")
@click.option("--lime",       is_flag=True, default=True, help="Use the LIME method. By default, the DUAL method is used.")
@click.option("--sigma",      type=int,     default=3,    help="Spatial standard deviation for spatial affinity based Gaussian weights.")
@click.option("--bc",         type=float,   default=1,    help="Parameter for controlling the influence of Mertens's contrast measure.")
@click.option("--bs",         type=float,   default=1,    help="Parameter for controlling the influence of Mertens's saturation measure.")
@click.option("--be",         type=float,   default=1,    help="Parameter for controlling the influence of Mertens's well exposedness measure.")
@click.option("--eps",        type=float,   default=1e-3, help="Constant to avoid computation instability.")
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
    gamma     : float,
    lambda_   : float,
    lime      : bool,
    sigma     : int,
    bc        : float,
    bs        : float,
    be        : float,
    eps       : float,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
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
        "fullname"  : fullname,
        "save_dir"  : save_dir,
        "device"    : device,
        "imgsz"     : imgsz,
        "gamma"     : gamma,
        "lambda_"   : lambda_,
        "lime"      : lime,
        "sigma"     : sigma,
        "bc"        : bc,
        "bs"        : bs,
        "be"        : be,
        "eps"       : eps,
        "resize"    : resize,
        "benchmark" : benchmark,
        "save_image": save_image,
        "verbose"   : verbose
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
