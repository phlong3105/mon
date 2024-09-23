#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/zkawfanx/StableLLVE

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torchvision
from PIL import Image

import mon
from model import UNet
from mon import RUN_DIR, ZOO_DIR

console = mon.console


def test(args: argparse.Namespace):
    args.input_dir  = mon.Path(args.input_dir)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.input_dir}")
    
    # Load model
    model = UNet(n_channels=3, bilinear=True)
    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    
    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = model,
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
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                image          = Image.open(image_path).convert("RGB")
                image          = (np.asarray(image) / 255.0)
                image          = torch.from_numpy(image).float()
                image          = image.permute(2, 0, 1)
                image          = image.cuda().unsqueeze(0)
                start_time     = time.time()
                enhanced_image = model(image)
                run_time       = (time.time() - start_time)
                output_path    = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
        
    """
    with torch.no_grad():
        for i, filename in enumerate(filenames):
            test = cv2.imread(filename)/255.0
            test = np.expand_dims(test.transpose([2,0,1]), axis=0)
            test = torch.from_numpy(test).to(device="cuda", dtype=torch.float32)
            out  = model(test)
            out  = out.to(device="cpu").numpy().squeeze()
            out  = np.clip(out*255.0, 0, 255)
            path = filename.replace('/test/','/results/')[:-4]+'.png'
            # folder = os.path.dirname(path)
            # if not os.path.exists(folder):
            #     os.makedirs(folder)
            cv2.imwrite(path, out.astype(np.uint8).transpose([1,2,0]))
            print('%d|%d'%(i+1, len(filenames)))
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str, default="./data/test/*")
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/utvnet")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/stablellve/stablellve-checkpoint.pth")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--benchmark",  action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
