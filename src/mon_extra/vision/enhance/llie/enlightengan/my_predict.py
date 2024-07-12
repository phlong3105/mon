#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/arsenyinfo/EnlightenGAN-inference
# pip install onnx-tool
# https://pypi.org/project/onnx-tool/0.1.7/

from __future__ import annotations

import argparse
import time

import cv2
import torch

import mon
from onnx_model import EnlightenOnnxModel

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    weights   = weights or mon.ZOO_DIR / "vision/enhance/llie/enlightengan/enlightengan/custom/enlightengan.onnx"
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Measure efficiency score
    # if benchmark:
    #     model  = EnlightenOnnxModel(weights=weights)
    #     inputs = {"input": create_ndarray_f32((1, 3, 512, 512)), }
    #     onnx_tool.model_profile(str(_current_dir/"enlighten_inference/enlighten.onnx"), inputs, None)
    
    # Model
    model = EnlightenOnnxModel(weights=weights)
    
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
                image_path     = meta["path"]
                image          = cv2.imread(str(image_path))
                start_time     = time.time()
                enhanced_image = model.predict(image)
                run_time       = (time.time() - start_time)
                output_path    = save_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                sum_time      += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

if __name__ == "__main__":
    args = mon.parse_predict_args()
    predict(args)
    
# endregion
