#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance

from __future__ import annotations

import argparse
import os

import numpy as np
import torch.optim
import torchvision
from IQA_pytorch import SSIM
from tqdm import tqdm

import mon
from data_loaders.exposure import exposure_loader
from model.IAT_main import IAT
from utils import PSNR

console = mon.console


def eval(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    console.log(f"Data: {args.data}")
    
    test_dataset = exposure_loader(
        images_path = str(args.data),
        mode        = "test",
        expert      = args.expert,
        normalize   = args.pre_norm
    )
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True
    )
    
    model = IAT(type="exp").cuda()
    model.load_state_dict(torch.load(str(args.weights)))
    model.eval()
    
    ssim      = SSIM()
    psnr      = PSNR()
    ssim_list = []
    psnr_list = []
    
    with torch.no_grad():
        for i, images in tqdm(enumerate(test_loader)):
            # print(i)
            low_img, high_img = images[0].cuda(), images[1].cuda()
            # print(low_img.shape)
            mul, add, enhanced_img = model(low_img)
            
            if args.save:
                result_path = args.output_dir / f"{i}.png"
                torchvision.utils.save_image(enhanced_img, str(result_path))
    
            ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
            psnr_value = psnr(enhanced_img, high_img).item()
    
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
    
    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print("The SSIM Value is:", SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,  default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/test/INPUT_IMAGES/")
    parser.add_argument("--weights",    type=str,  default="best_Epoch_exposure.pth")
    parser.add_argument("--image-size", type=int,  default=512)
    parser.add_argument("--save",       type=bool, default=False)
    parser.add_argument("--expert",     type=str,  default="c")  # Choose the evaluation expert
    parser.add_argument("--pre-norm",   type=bool, default=False)
    parser.add_argument("--gpu",        type=str,  default=3)
    parser.add_argument("--output-dir", type=str,  default=mon.RUN_DIR/"predict/iat")
    args = parser.parse_args()
    
    args.data       = mon.Path(args.data)
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    eval(args)
