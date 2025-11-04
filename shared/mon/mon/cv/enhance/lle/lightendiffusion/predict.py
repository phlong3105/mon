#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LightenDiffusion model prediction pipeline for low-light image enhancement.

References:
    - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
      Latent-Retinex Diffusion Models," ECCV 2024.
    - Code: https://github.com/JianghaiSCU/LightenDiffusion
"""

import argparse
import copy

import box
import numpy as np
import torch
import yaml

import lightendiffusion
import mon
from mon import albumentations as A
from mon.core.nn import functional as F

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


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
    cfg_path = root_dir / "lightendiffusion" / "option" / args.cfg
    with open(str(cfg_path), "r") as f:
        cfgs = yaml.safe_load(f)
    cfgs = dict2namespace(cfgs)

    # Start
    mon.rt.print_run_summary(args)

    # Device
    device      = mon.create_device(args.device)
    cfgs.device = device

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
    model_args = argparse.Namespace(**{
        "mode"        : "evaluation",
        "resume"      : str(pretrained),
        "image_folder": str(args.save_dir),
    })
    diffusion = lightendiffusion.LightenDiffusion(model_args, cfgs)
    diffusion.load_ddm_ckpt(str(pretrained), ema=False)
    diffusion.model.eval()

    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(diffusion.model)
    
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
            meta     = datapoint["meta"][0]
            path     = mon.Path(meta["path"])
            h0, w0   = mon.image.imgsz(meta["orig_shape"])
            image    = datapoint["image"]
            img_h_64 = int(64 * np.ceil(h0 / 64.0))
            img_w_64 = int(64 * np.ceil(w0 / 64.0))
            x_cond   = F.pad(image, (0, img_w_64 - w0, 0, img_h_64 - h0), "reflect")
            x_cond   = x_cond.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = diffusion.model(torch.cat((x_cond, x_cond), dim=1))["pred_x"]
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs[:, :, :h0, :w0]
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
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
