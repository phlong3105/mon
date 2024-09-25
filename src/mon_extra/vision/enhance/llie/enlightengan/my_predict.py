#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/arsenyinfo/EnlightenGAN-inference
# pip install onnx-tool
# https://pypi.org/project/onnx-tool/0.1.7/

from __future__ import annotations

import argparse
from typing import Sequence

import cv2
import numpy as np
import torch

import mon
from mon import nn
from onnx_model import EnlightenOnnxModel

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def compute_efficiency_score(
    model     : nn.Module,
    image_size: int | Sequence[int] = 512,
    channels  : int  = 3,
    runs      : int  = 1000,
    use_cuda  : bool = True,
    verbose   : bool = False,
):
    # Define input tensor
    h, w  = mon.get_image_size(image_size)
    input = np.random.rand(h, w, channels)
    
    # Get time
    timer = mon.Timer()
    for i in range(runs):
        timer.tick()
        _ = model.predict(input)
        timer.tock()
    avg_time = timer.avg_time
    
    # Print
    if verbose:
        # console.log(f"FLOPs (G) : {flops:.4f}")
        # console.log(f"Params (M): {params:.4f}")
        console.log(f"Time (s)  : {avg_time:.17f}")
    

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    weights      = weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Model
    model = EnlightenOnnxModel(weights=weights)
    
    # Measure efficiency score
    if benchmark:
        compute_efficiency_score(
            model      = model,
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = True,
        )
        # model  = EnlightenOnnxModel(weights=weights)
        # inputs = {"input": create_ndarray_f32((1, 3, 512, 512)), }
        # onnx_tool.model_profile(str(current_dir/"enlighten_inference/enlighten.onnx"), inputs, None)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint.get("image")
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                
                # Infer
                timer.tick()
                enhanced_image = model.predict(image)
                timer.tock()
                
                # Post-processing
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), enhanced_image)
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
