#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time

import torchvision

import mon
from modeling import model
from option import *
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # For GPU only
device = get_device()

console = mon.console

class Tester:
    
    def __init__(self):
        self.scale_factor = 12
        self.net          = model.enhance_net_nopool(self.scale_factor, conv_type="dsc").to(device)
        self.net.load_state_dict(torch.load(args.weights, map_location=device))

    def inference(self, image_path):
        # Read image from path
        data_lowlight = image_from_path(str(image_path))

        # Scale image to have the resolution of multiple of 4
        data_lowlight = scale_image(data_lowlight, self.scale_factor, device) if self.scale_factor != 1 else data_lowlight

        # Run model inference
        start_time = time.time()
        enhanced_image, params_maps = self.net(data_lowlight)
        run_time   = (time.time() - start_time)

        # Load result directory and save image
        # result_path = os.path.join(args.test_dir, os.path.relpath(image_path, args.input_dir))
        # os.makedirs(os.path.dirname(result_path), exist_ok=True)
        # torchvision.utils.save_image(enhanced_image, result_path)
        
        return enhanced_image, run_time

    def test(self):
        self.net.eval()
        
        console.log(f"Data: {args.input_dir}")
        args.input_dir = mon.Path(args.input_dir)
        image_paths    = list(args.input_dir.rglob("*"))
        image_paths    = [p for p in image_paths if p.is_image_file()]
        
        args.output_dir = mon.Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Measure efficiency score
        if args.benchmark:
            h = (args.image_size // self.scale_factor) * self.scale_factor
            w = (args.image_size // self.scale_factor) * self.scale_factor
            flops, params, avg_time = mon.compute_efficiency_score(
                model      = self.net,
                image_size = [h, w],
                channels   = 3,
                runs       = 1000,
                use_cuda   = True,
                verbose    = False,
            )
            console.log(f"FLOPs  = {flops:.4f}")
            console.log(f"Params = {params:.4f}")
            console.log(f"Time   = {avg_time:.17f}")
        
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                enhanced_image, run_time = self.inference(image_path)
                sum_time    += run_time
                output_path  = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


if __name__ == "__main__":
    t = Tester()
    t.test()
