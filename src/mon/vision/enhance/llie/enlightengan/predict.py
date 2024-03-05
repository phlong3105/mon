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
from enlighten_inference import EnlightenOnnxModel
from mon import ZOO_DIR, RUN_DIR

console      = mon.console
_current_dir = mon.Path(__file__).absolute().parent


def predict(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    # Load model
    model = EnlightenOnnxModel()
    
    # Measure efficiency score
    if args.benchmark:
        inputs = {"input": create_ndarray_f32((1, 3, 512, 512)), }
        onnx_tool.model_profile(str(_current_dir/"enlighten_inference/enlighten.onnx"), inputs, None)

    #
    with torch.no_grad():
        image_paths = list(args.input_dir.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                image          = cv2.imread(str(image_path))
                start_time     = time.time()
                enhanced_image = model.predict(image)
                run_time       = (time.time() - start_time)
                result_path    = args.output_dir / image_path.name
                cv2.imwrite(str(result_path), enhanced_image)
                sum_time      += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str,  default="./data/test/*")
    parser.add_argument("--output-dir", type=str,  default=RUN_DIR / "predict/vision/enhance/llie/enlightengan")
    parser.add_argument("--weights",    type=str,  default=ZOO_DIR / "vision/enhance/llie/enlightengan/enlightengan.onnx")
    parser.add_argument("--image-size", type=int,  default=512)
    parser.add_argument("--benchmark",  action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)