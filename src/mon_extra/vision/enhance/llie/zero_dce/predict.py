#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model
import mon
from mon import RUN_DIR, ZOO_DIR

console = mon.console


def run_infer(image_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_lowlight = Image.open(image_path).convert("RGB")
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net    = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(args.weights))
    start_time = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)
    run_time   = (time.time() - start_time)
    # print(run_time)
    '''
    image_path  = image_path.replace("test_data", "result")
    result_path = image_path
    if not os.path.exists(image_path.replace("/" + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace("/" + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(enhanced_image, result_path)
    '''
    return enhanced_image, run_time


def test(args):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(args.weights))
    
    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = DCE_net,
            image_size = args.image_size,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    #
    with torch.no_grad():
        image_paths = list(args.input_dir.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        image_paths.sort()
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                enhanced_image, run_time = run_infer(image_path)
                output_path = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, default="data/test_data/")
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/zerodce")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/zerodce/best.pth")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--benchmark",  action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
