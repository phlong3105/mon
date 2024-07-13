#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import socket
import time

import click
import torch
import torch.nn.functional as F
from skimage.util import img_as_ubyte

import mon
import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Override options with args
    # gpu_list = ",".join(str(x) for x in args.gpus)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    opt           = parse(args.opt, is_train=False)
    opt["dist"]   = False
    opt["device"] = device
    
    # Model
    model      = create_model(opt).net_g
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["params"])
    except:
        new_checkpoint = {}
        for k in checkpoint["params"]:
            new_checkpoint["module." + k] = checkpoint["params"][k]
        model.load_state_dict(new_checkpoint)
    
    print("===>Testing using weights: ", weights)
    model.to(device)
    # model = nn.DataParallel(model)
    model.eval()
    
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
        factor   = 4
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                image_path = meta["path"]
                
                if resize:
                    h0, w0 = mon.get_image_size(images)
                    images = mon.resize(images, imgsz)
                    console.log("Resizing images to: ", images.shape[2], images.shape[3])
                    # images = proc.resize(input=images, size=[1000, 666])
                
                # Padding in case images are not multiples of 4
                h, w  = images.shape[2], images.shape[3]
                H, W  = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh  = H - h if h % factor != 0 else 0
                padw  = W - w if w % factor != 0 else 0
                input = F.pad(images, (0, padw, 0, padh), 'reflect')
                input = input.to(device)
                
                start_time = time.time()
                restored = model(input)
                run_time = (time.time() - start_time)
                
                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]
                if resize:
                    restored = mon.resize(restored, (h0, w0))
                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                
                output_path = save_dir / image_path.name
                utils.save_img(str(output_path), img_as_ubyte(restored))
                sum_time += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
       
# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
