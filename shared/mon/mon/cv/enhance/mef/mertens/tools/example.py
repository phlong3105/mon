#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import box
import cv2
import numpy as np
import torch

import mon
from mon import Path
from ..mertens import Mertens

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]
data_dir     = root_dir / "data"
run_dir      = root_dir / "run"


def run(args: dict | box.Box):
    image_dir = args.image_dir
    image_dir = data_dir / image_dir
    out_path  = run_dir  / "sample" / f"{image_dir.stem}.jpg"
    
    model  = Mertens()
    
    timers = mon.TimeProfiler()
    timers.total.tick()
    
    # Preprocess
    timers.preprocess.tick()
    image_files = sorted(list(image_dir.rglob("*")))
    images      = [cv2.imread(str(f)) for f in image_files]
    images      = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images      = torch.Tensor(np.array(images))
    images      = images / 255.0  # Normalize to [0, 1]
    images      = images.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
    timers.preprocess.tock()
    
    # Inference
    timers.infer.tick()
    # fused = mertens.mertens(images)
    fused = model(images)
    timers.infer.tock()
    
    # Postprocess
    timers.postprocess.tick()
    # fused = mon.image.to_array(fused)
    timers.postprocess.tock()
    
    # Save
    mon.image.save_image(fused, out_path)
    timers.total.tock()

    # Finish
    timers.print()


# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="house", help="Path to image folder")
    args = parser.parse_args()
    run(args)
    

if __name__ == "__main__":
    main()
