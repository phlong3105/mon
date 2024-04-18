#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/wyf0912/LLFlow

from __future__ import annotations

import argparse
import copy
import os
import socket
import time

import click
import cv2
import numpy as np
import torch

import mon
import options.options as option
from models import create_model

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t):
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape
    h_start = max(h // 2 - size // 2, 0)
    h_end   = min(h_start + size, h)
    w_start = max(w // 2 - size // 2, 0)
    w_end   = min(w_start + size, w)
    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], "reflect")


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH        = cv2.equalizeHist(b)
    gH        = cv2.equalizeHist(g)
    rH        = cv2.equalizeHist(r)
    result    = cv2.merge((bH, gH, rH))
    return result


def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C
    h, w, _ = img.shape
    h1, w1  = (times - h % times) // 2, (times - w % times) // 2
    h2, w2  = (times - h % times) - h1, (times - w % times) - w1
    img     = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


def predict(args: argparse.Namespace):
    # General config
    opt       = args.opt
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    device    = args.device
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    opt            = option.parse(opt, is_train=False)
    opt["gpu_ids"] = None
    opt            = option.dict_to_nonedict(opt)
    
    # Device
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Model
    model          = create_model(opt)
    # model_path     = opt_get(opt, ["model_path"], None)
    model.load_network(load_path=weights, network=model.netG)
    model.netG     = model.netG.to(device)
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = copy.deepcopy(model).measure_efficiency_score(image_size=imgsz)
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
                lr         = imread(str(image_path))
                raw_shape  = lr.shape
                
                start_time = time.time()
                lr, padding_params = auto_padding(lr)
                his = hiseq_color_cv2_img(lr)
                if opt.get("histeq_as_input", False):
                    lr = his
                lr_t = t(lr)
                if opt["datasets"]["train"].get("log_low", False):
                    lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
                if opt.get("concat_histeq", False):
                    his  = t(his)
                    lr_t = torch.cat([lr_t, his], dim=1)
                heat = opt["heat"]
                
                with torch.cuda.amp.autocast():
                    sr_t = model.get_sr(lq=lr_t.to(device), heat=None)
                sr = rgb(
                    torch.clamp(sr_t, 0, 1)[
                        :, :,
                        padding_params[0]:sr_t.shape[2] - padding_params[1],
                        padding_params[2]:sr_t.shape[3] - padding_params[3]
                    ]
                )
                # assert raw_shape == sr.shape
                run_time  = (time.time() - start_time)
                sum_time += run_time
                
                output_path = save_dir / image_path.name
                imwrite(str(output_path), sr)
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
    config = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = weights or mon.ZOO_DIR / "vision/enhance/llie/llflow/llflow_lol_v1.pth"
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)[0]
    
    # Update arguments
    args = {
        "root"      : root,
        "config"    : config,
        "opt"       : config,
        "weights"   : weights,
        "model"     : model,
        "data"      : data,
        "fullname"  : fullname,
        "save_dir"  : save_dir,
        "device"    : device,
        "imgsz"     : imgsz,
        "resize"    : resize,
        "benchmark" : benchmark,
        "save_image": save_image,
        "verbose"   : verbose,
        "seed"      : 2,
    }
    args = argparse.Namespace(**args)
    
    predict(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
