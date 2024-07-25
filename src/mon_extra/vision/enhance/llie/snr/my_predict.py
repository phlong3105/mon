#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

import config.options as option
import data.util as dutil
import mon
import utils.util as util
from models import create_model

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    weights   = args.weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
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
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path = meta["path"]
                image      = dutil.read_img(None, str(image_path))
                image      = image[:, :, ::-1]
                h, w       = mon.get_image_size(image)
                # image      = cv2.resize(image, (600, 400))
                image      = mon.resize_divisible(image, 32)
                image_nf   = cv2.blur(image, (5, 5))
                image_nf   = image_nf * 1.0 / 255.0
                image_nf   = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image      = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image      = image.unsqueeze(0).to(device)
                image_nf   = image_nf.unsqueeze(0).to(device)
                
                timer.tick()
                model.feed_data(
                    data = {
                        "idx": meta["id"],
                        "LQs": image,
                        "nf" : image_nf,
                    },
                    need_GT=False
                )
                model.test()
                timer.tock()
                
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals["rlt"])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                output_path    = save_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                # torchvision.utils.save_image(enhanced_image, str(output_path))
        # avg_time = float(timer.total_time / len(data_loader))
        avg_time   = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")
    
# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
