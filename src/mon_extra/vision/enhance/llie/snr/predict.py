#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

import data.util as dutil
import mon
import options.options as option
import utils.util as util
from models import create_model
from mon import RUN_DIR, ZOO_DIR

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    args.opt = option.parse(args.opt, is_train=False)
    args.opt = option.dict_to_nonedict(args.opt)
    args.opt["path"]["pretrain_model_G"] = str(args.weights)
    
    console.log(f"Data: {args.input_dir}")
    
    # Load model
    model = create_model(args.opt)
    
    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = model.measure_efficiency_score(image_size=args.image_size)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")

    #
    with torch.no_grad():
        image_paths = list(args.input_dir.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for i, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                image    = dutil.read_img(None, str(image_path))
                image    = image[:, :, ::-1]
                h, w, c  = image.shape
                image    = cv2.resize(image, (600, 400))
                image_nf = cv2.blur(image, (5, 5))
                image_nf = image_nf * 1.0 / 255.0
                image_nf = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
                image    = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
                image    = image.unsqueeze(0).cuda()
                image_nf = image_nf.unsqueeze(0).cuda()
                data     = {
                    "idx": i,
                    "LQs": image,
                    "nf" : image_nf,
                }
                start_time = time.time()
                model.feed_data(data, need_GT=False)
                model.test()
                run_time   = (time.time() - start_time)
                
                visuals        = model.get_current_visuals(need_GT=False)
                enhanced_image = util.tensor2img(visuals['rlt'])  # uint8
                enhanced_image = cv2.resize(enhanced_image, (w, h))
                output_path    = args.output_dir / image_path.name
                cv2.imwrite(str(output_path), enhanced_image)
                # torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
 
# endregion


# region Main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, default="data/test_data/")
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/snr")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/snr/snr-lolv1.pth")
    parser.add_argument("--opt",        type=str, default="./options/test/LOLv1.yml", help="Path to options YAML file.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--benchmark",  action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)

# endregion
