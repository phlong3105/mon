#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourierDiff model for zero-shot joint low-light enhancement and deblurring.

References:
    - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
      Enhancement and Deblurring," CVPR 2024.
    - Code: https://github.com/aipixel/FourierDiff
"""

import argparse
import copy

import box
import torch
import yaml

import fourierdiff
import mon
from mon import albumentations as A

mon.dev()
torch.set_printoptions(sci_mode=False)

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = root_dir / "fourierdiff" / "option" / args.cfg
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = dict2namespace(cfg)

    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device     = mon.create_device(args.device)
    cfg.device = device

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    runner = fourierdiff.FourierDiff(dict2namespace(args), cfg)

    # Benchmark
    # if args.benchmark:
    #     mon.nn.benchmark(model)
    
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
    runner.sample(
        pretrained,
        data_name,
        dataloader,
        args.imgsz,
        args.resize,
        args.save_image,
        args.save_dir,
        args.keep_subdirs,
        args.save_nearby,
        timers
    )
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
