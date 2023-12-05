#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/CharlieZCJ/UTVNet

'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}

Tested with Pytorch 1.7.1, Python 3.6

Author: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torchvision
from PIL import Image

import mon
from models import network
from mon import RUN_DIR, ZOO_DIR

console = mon.console


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str, default="data/test_data/")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/utvnet/utvnet-model_test.pt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/utvnet")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    console.log(f"Data: {args.data}")
    
    # Load model
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    model  = network.UTVNet().to(device)
    model.load_state_dict(torch.load(str(args.weights), map_location=device))
    
    # Measure efficiency score
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
    
    #
    torch.set_grad_enabled(False)
    with torch.no_grad():
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
                image          = Image.open(image_path).convert("RGB")
                image          = (np.asarray(image) / 255.0)
                image          = torch.from_numpy(image).float()
                image          = image.permute(2, 0, 1)
                image          = image.cuda().unsqueeze(0)
                start_time     = time.time()
                enhanced_image = model(image)
                enhanced_image = enhanced_image.clamp(0, 1).cpu()
                run_time       = (time.time() - start_time)
                result_path    = args.output_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(result_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")
        
    """
    if args.data_name == 'sRGBSID':
        test_input_dir = './dataset/sRGBSID/test/1/'
        test_input_dir2 = './dataset/sRGBSID/test/2/'
        test_gt_dir = './dataset/sRGBSID/gt/test/'
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '512', args.data_name)

    else:
        test_input_dir = './dataset/ELD/{}/'.format(args.data_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/ELD/{}g/'.format(args.data_name)
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '1024', args.data_name)

    test(model, args, loaderTest, device)
    """


if __name__ == "__main__":
    test()
