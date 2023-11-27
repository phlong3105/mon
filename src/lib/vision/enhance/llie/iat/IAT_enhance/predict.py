#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import Normalize

import mon
from model.IAT_main import IAT
from mon import ZOO_DIR

console = mon.console


def predict(args):
    console.log(f"Data: {args.data}")
    
    # Load Pre-train Weights
    model = IAT().cuda()
    if args.task == "exposure":
        model.load_state_dict(torch.load(str(args.exposure_weights)))
    elif args.task == "enhance":
        model.load_state_dict(torch.load(str(args.enhance_weights)))
    else:
        warnings.warn("Only could be `exposure` or `enhance`")
    model.eval()
    
    # Measure efficiency score
    '''
    flops, params, avg_time = mon.calculate_efficiency_score(
        model      = model,
        image_size = args.image_size,
        channels   = 3,
        runs       = 100,
        use_cuda   = True,
        verbose    = False,
    )
    console.log(f"FLOPs  = {flops:.4f}")
    console.log(f"Params = {params:.4f}")
    console.log(f"Time   = {avg_time:.4f}")
    '''
    
    #
    normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    image_paths = list(args.data.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for _, image_path in pbar.track(
            sequence    = enumerate(image_paths),
            total       = len(image_paths),
            description = f"[bright_yellow] Inferring"
        ):
            # console.log(image_path)
            image = Image.open(image_path)
            image = (np.asarray(image) / 255.0)
            if image.shape[2] == 4:
                image = image[:, :, :3]
            input = torch.from_numpy(image).float().cuda()
            input = input.permute(2, 0, 1).unsqueeze(0)
            if args.normalize:  # False
                input = normalize_process(input)
            # Forward Network
            start_time = time.time()
            _, _ , enhanced_image = model(input)
            run_time = (time.time() - start_time)
            # console.log(run_time)
            result_path = args.output_dir / image_path.name
            torchvision.utils.save_image(enhanced_image, str(result_path))
            sum_time += run_time
    avg_time = float(sum_time / len(image_paths))
    console.log(f"Average time: {avg_time}")
    
    """
    image = Image.open(args.file_name)
    image = (np.asarray(image) / 255.0)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    input = torch.from_numpy(image).float().cuda()
    input = input.permute(2, 0, 1).unsqueeze(0)
    if args.normalize:  # False
        input = normalize_process(input)
    
    # Forward Network
    _, _ , enhanced_img = model(input)
    
    torchvision.utils.save_image(enhanced_img, "result.png")
    """
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             type=str,  default="demo_imgs/low_demo.jpg")
    parser.add_argument("--exposure-weights", type=str,  default=ZOO_DIR/"vision/enhance/llie/iat-exposure.pth")
    parser.add_argument("--enhance-weights",  type=str,  default=ZOO_DIR/"vision/enhance/llie/iat-lol-v1.pth")
    parser.add_argument("--image-size",       type=int,  default=512)
    parser.add_argument("--normalize",        action="store_true", default=False)
    parser.add_argument("--task",             type=str,  default="enhance", help="Choose from exposure or enhance")
    parser.add_argument("--output-dir",       type=str,  default=mon.RUN_DIR/"predict/iat")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        predict(args)
