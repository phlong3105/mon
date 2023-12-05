#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time

import mon
from model import RetinexNet
from mon import ZOO_DIR, RUN_DIR

console = mon.console


def predict(args, model):
    image_paths = list(args.data.rglob("*"))
    image_paths = [path for path in image_paths if path.is_image_file()]
    image_paths.sort()
    # print('Number of evaluation images: %d' % len(test_low_data_names))
    
    start_time = time.time()
    model.predict(image_paths, res_dir=args.output_dir, ckpt_dir=args.weights)
    run_time   = (time.time() - start_time)
    avg_time   = float(run_time / len(image_paths))
    console.log(f"Average time: {avg_time}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default="./data/test/low/", help="directory storing the test data")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/retinexnet", help="directory for checkpoints")
    parser.add_argument("--gpu",        type=str, default="0", help="GPU ID (-1 for CPU)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/retinexnet", help="directory for saving the results")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Data: {args.data}")

    if args.gpu != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
       
        # Create the model
        model = RetinexNet(args.image_size).cuda()

        # Test the model
        predict(args, model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
