#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance

from __future__ import annotations

import argparse
import os
import socket
import time

import click
import cv2
import numpy as np
import torch

import data.util as dutil
import config.options as option
import utils.util as util
from models import create_model
from mon import core, data as d

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = core.Path(args.save_dir)
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Override options with args
    opt           = option.parse(args.opt, is_train=False)
    opt           = option.dict_to_nonedict(opt)
    opt["device"] = device
    
    # Load model
    opt["path"]["pretrain_model_G"] = str(weights)
    model = create_model(opt)
    
    # Measure efficiency score
    if benchmark:
        flops, params, avg_time = model.measure_efficiency_score(image_size=imgsz)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
        
    # Predicting
    '''
    data = mon.DATA_DIR / opt["datasets"]["test"]["dataroot_LQ"]
    with torch.no_grad():
        image_paths = list(data.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for i, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Predicting"
            ):
                image    = dutil.read_img(None, str(image_path))
                image    = image[:, :, ::-1]
                h, w, c  = image.shape
                image    = cv2.resize(image, (600, 400))
                image_nf = cv2.blur(image, (5, 5))
                image_nf = image_nf * 1.0 / 255.0
                image_nf = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image    = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image    = image.unsqueeze(0).cuda()
                image_nf = image_nf.unsqueeze(0).cuda()
                
                start_time = time.time()
                model.feed_data(
                    data = {
                        "idx": i,
                        "LQs": image,
                        "nf" : image_nf,
                    },
                    need_GT=False
                )
                model.test()
                run_time   = (time.time() - start_time)
                
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals["rlt"])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                output_path    = save_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                # torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
    '''
    
    # Data I/O
    console.log(f"{data}")
    data_name, data_loader, data_writer = d.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting 2
    with torch.no_grad():
        sum_time = 0
        with core.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path = meta["path"]
                image      = dutil.read_img(None, str(image_path))
                image      = image[:, :, ::-1]
                h, w, c    = image.shape
                image      = cv2.resize(image, (600, 400))
                image_nf   = cv2.blur(image, (5, 5))
                image_nf   = image_nf * 1.0 / 255.0
                image_nf   = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image      = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image      = image.unsqueeze(0).to(device)
                image_nf   = image_nf.unsqueeze(0).to(device)
                
                start_time = time.time()
                model.feed_data(
                    data = {
                        "idx": meta["id"],
                        "LQs": image,
                        "nf" : image_nf,
                    },
                    need_GT=False
                )
                model.test()
                run_time   = (time.time() - start_time)
                
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals["rlt"])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                output_path    = save_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                # torchvision.utils.save_image(enhanced_image, str(output_path))
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
    config   = core.parse_config_file(project_root=_current_dir / "config", config=config)
    
    # Parse arguments
    root     = core.Path(root)
    weights  = core.to_list(weights)
    project  = root.name
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = core.parse_device(device)
    imgsz    = core.parse_hw(imgsz)[0]
    
    # Update arguments
    args = {
        "root"      : root,
        "config"    : config,
        "opt"       : config,
        "weights"   : weights,
        "model"     : model,
        "data"      : data,
        "project"   : project,
        "fullname"  : fullname,
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
