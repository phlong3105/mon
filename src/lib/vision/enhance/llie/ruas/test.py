#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/KarelZhang/RUAS

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image
from torch.autograd import Variable

import mon
from model import Network
from mon import DATA_DIR, ZOO_DIR, RUN_DIR
from multi_read_data import MemoryFriendlyLoader

console = mon.console


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


def test(args):
    if not torch.cuda.is_available():
        console.log("No gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled   = True
    torch.cuda.manual_seed(args.seed)
    # print("GPU device = %d" % args.gpu)
    # print("args = %s", args)
    console.log(f"Data: {args.data}")
    
    model      = Network()
    model      = model.cuda()
    model_dict = torch.load(str(args.weights))
    model.load_state_dict(model_dict)
    for p in model.parameters():
        p.requires_grad = False
    
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
    
    # Prepare DataLoader
    # test_low_data_names = r"H:\CVPR2021\LOL-700\input-100/*.png"
    # test_low_data_names = r"H:\image-enhance\LLIECompared\DarkFace1000\input/*.png"
    test_low_data_names = args.data
    test_dataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task="test")
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size  = 1,
        pin_memory  = True,
        num_workers = 0
    )
    
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for _, (input, image_name) in pbar.track(
                sequence    = enumerate(test_loader),
                total       = len(test_loader),
                description = f"[bright_yellow] Predicting"
            ):
                input          = Variable(input, volatile=True).cuda()
                image_name     = image_name[0].split(".")[0]
                u_name         = f"{image_name}.png"
                # console.log(f"Processing {u_name}")

                start_time     = time.time()
                u_list, r_list = model(input)
                run_time       = (time.time() - start_time)
                sum_time      += run_time
                
                save_images(u_list[-1], str(args.output_dir / u_name))
                # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                """
                if args.model == "lol":
                    save_images(u_list[-1], u_path)
                elif args.model == "upe" or args.model == "dark":
                    save_images(u_list[-2], u_path)
                """
        avg_time = sum_time / len(test_dataset)
        console.log(f"Average time: {avg_time}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ruas")
    parser.add_argument("--data",       type=str, default=DATA_DIR / "lol")
    parser.add_argument("--weights",    type=str, default=ZOO_DIR / "vision/enhance/llie/ruas/ruas-lol.pt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--seed",       type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=RUN_DIR / "predict/vision/enhance/llie/ruas")
    args = parser.parse_args()
    
    args.output_dir = mon.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # (args.output_dir / "lol").mkdir(parents=True, exist_ok=True)
    # (args.output_dir / "dark").mkdir(parents=True, exist_ok=True)
    
    test(args)
