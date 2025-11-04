#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZERO-IG model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
      Enhancement for Low-Light Images," CVPR 2024.
    - Code: https://github.com/Doyle59217/ZeroIG
"""

import copy
import logging
import sys

import box
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from thop import profile
from torch.autograd import Variable

import mon
import zeroig
from mon import albumentations as A

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def save_images(tensor):
    image_numpy = tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    flops, _           = profile(model, inputs=(input_tensor,))
    flops_in_gigaflops = flops / 1e9  # Convert FLOPs to gigaflops (G)
    return flops_in_gigaflops


def benchmark():
    model = zeroig.ZERO_IG()
    # flops, params = metric.compute_efficiency_score(model=model)
    total_params  = calculate_model_parameters(model)
    # mon.log(f"FLOPs     : {flops:.4f}")
    # mon.log(f"Params    : {params:.4f}")
    mon.log(f"Total Params = {total_params:.4f}")


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
        cudnn.enabled   = True
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        logging.info("no gpu device available")
        sys.exit(1)

    # Seed
    mon.set_random_seed(args.seed)

    # Benchmark
    if args.benchmark:
        benchmark()
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            image  = image.to(device)
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            model = zeroig.ZERO_IG()
            model.enhance.in_conv.apply(model.enhance_weights_init)
            model.enhance.conv.apply(model.enhance_weights_init)
            model.enhance.out_conv.apply(model.enhance_weights_init)
            model = model.to(device)
            model.train()
            optimizer = mon.nn.Adam(model.parameters(), **args.optimizer)
            input     = Variable(image, requires_grad=False).to(device)
            for _ in range(args.epochs):
                optimizer.zero_grad()
                optimizer.param_groups[0]["capturable"] = True
                loss = model._loss(input)
                loss.backward()
                mon.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model = zeroig.ZERO_IG_Finetune(model.state_dict()).to(device)
            input = Variable(image).to(device)
            outputs = model(input)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced, denoise = outputs
            # enhanced = save_images(enhanced)
            # denoise  = save_images(denoise)
            enhanced = mon.image.to_array(enhanced)
            denoise  = mon.image.to_array(denoise)
            h1, w1  = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
                denoise  = cv2.resize(denoise, (w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(denoise, out_path)

            # if args.save_debug:
            #    out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, f"{mon.SAVE_IMAGE_DIR}_denoise", path, args.keep_subdirs, args.save_nearby)
            #    out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
            #    mon.image.save_image(denoise, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
