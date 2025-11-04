#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.optim

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
from net.model import AirNet
from utils.image_io import save_image_tensor

console      = console
current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.create_device(args.device)
    epochs       = args.epochs
    imgsz        = args.imgsz[0]
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    mode         = args.mode
    batch_size   = args.batch_size
    opt          = argparse.Namespace(
        **{
            "mode"      : mode,
            "batch_size": batch_size,
        }
    )
    
    # Model
    model = AirNet(opt)
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    
    # Benchmark
    if benchmark:
        params, macs, flops = mon.nn.compute_model_stats(
            model      = model,
            imgsz= imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        mon.log(f"FLOPs    : {flops:.4f}")
        mon.log(f"Params    : {params:.4f}")
        mon.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    mon.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            image       = datapoint["image"]
            meta        = datapoint["meta"]
            image_path  = Path(meta["path"])
            timer.tick()
            restored    = model(x_query=image, x_key=image)
            timer.tock()
            
            # Save
            if save_image:
                output_dir  = mon.rt.parse_output_dir(save_dir, data_name, mon.SAVE_IMAGE_DIR, image_path, keep_subdirs, save_nearby)
                output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_image_tensor(restored, output_path)
   
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
