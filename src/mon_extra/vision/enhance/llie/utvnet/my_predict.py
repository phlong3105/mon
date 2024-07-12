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
import copy
import time

import numpy as np
import torch
import torchvision
from PIL import Image

import mon
from models import network

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
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Model
    model = network.UTVNet().to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    torch.set_grad_enabled(False)
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path     = meta["path"]
                image          = Image.open(image_path).convert("RGB")
                image          = (np.asarray(image) / 255.0)
                image          = torch.from_numpy(image).float()
                image          = image.permute(2, 0, 1)
                image          = image.to(device).unsqueeze(0)
                start_time     = time.time()
                enhanced_image = model(image)
                enhanced_image = enhanced_image.clamp(0, 1).cpu()
                run_time       = (time.time() - start_time)
                output_path    = save_dir / image_path.name
                torchvision.utils.save_image(enhanced_image, str(output_path))
                sum_time      += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
        
    """
    if args.input_dir_name == 'sRGBSID':
        test_input_dir = './dataset/sRGBSID/test/1/'
        test_input_dir2 = './dataset/sRGBSID/test/2/'
        test_gt_dir = './dataset/sRGBSID/gt/test/'
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '512', args.input_dir_name)

    else:
        test_input_dir = './dataset/ELD/{}/'.format(args.input_dir_name)
        test_input_dir2 = ''
        test_gt_dir = './dataset/ELD/{}g/'.format(args.input_dir_name)
        loaderTest = dataset.rgbDataset(test_input_dir, test_input_dir2, test_gt_dir, 'test', '1024', args.input_dir_name)

    test(model, args, loaderTest, device)
    """

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args()
    args.weights = args.weights or mon.ZOO_DIR / "vision/enhance/llie/utvnet/utvnet/srgbsid/utvnet_srgbsid_pretrained.pt"
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
