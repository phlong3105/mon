#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import config.options as option
import cv2
import data.util as dutil
import numpy as np
import torch
import utils.util as util
from models import create_model

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = Path(args.save_dir)
    weights      = args.weights
    device       = mon.create_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    opt_path     = str(root_dir / "model_config" / args.opt_path)
    
    # Override options with args
    opt           = option.parse(opt_path, is_train=False)
    opt           = option.dict_to_nonedict(opt)
    opt["device"] = device
    
    # Load model
    opt["path"]["pretrain_model_G"] = str(weights)
    model = create_model(opt)
    
    # Measure efficiency score
    if benchmark:
        flops, params, avg_time = model.compute_model_stats()
        mon.log(f"FLOPs    : {flops:.4f}")
        mon.log(f"Params    : {params:.4f}")
        mon.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    mon.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow]Predicting"
            ):
                # Input
                meta       = datapoint["meta"]
                image_path = Path(meta["path"])
                image      = dutil.read_img(None, str(image_path))
                # image      = image[:, :, ::-1]
                h, w       = mon.image.imgsz(image)
                # image      = cv2.resize(image, (600, 400))
                image      = mon.resize(image, divisible_by=32)
                image_nf   = cv2.blur(image, (5, 5))
                image_nf   = image_nf * 1.0 / 255.0
                image_nf   = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image      = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image      = image.unsqueeze(0).to(device)
                image_nf   = image_nf.unsqueeze(0).to(device)
                
                # Infer
                timer.tick()
                model.feed_data(
                    data = {
                        "idx": i,
                        "LQs": image,
                        "nf" : image_nf,
                    },
                    need_GT=False
                )
                model.test()
                timer.tock()
                
                # Post-processing
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals["rlt"])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                
                # Save
                if save_image:
                    output_dir  = mon.rt.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                    output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), enhanced_image)
                    # torchvision.utils.save_image(enhanced_image, str(output_path))
        
        avg_time = float(timer.avg_time)
        mon.log(f"Average time: {avg_time}")
    

# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
