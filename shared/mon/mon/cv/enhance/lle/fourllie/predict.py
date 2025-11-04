#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourLLIE model prediction pipeline for low-light image enhancement.

References:
    - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
      Information," ACMMM 2023.
    - Code: https://github.com/wangchx67/FourLLIE
"""

import copy

import box
import cv2
import numpy as np
import torch

import fourllie
import mon
from fourllie import option, tensor2img
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = root_dir / "fourllie" / "option" / "test" / args.cfg
    cfgs     = option.parse(str(cfg_path), is_train=False)
    cfgs     = option.dict_to_nonedict(cfgs)
    
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
        # cfgs["path"]["pretrain_model_G"] = str(pretrained)
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = fourllie.FourLLIE(cfgs, weights=pretrained)
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
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
            image    = datapoint["image"][0]
            image_nf = cv2.blur(image, (5, 5))
            image_nf = image_nf * 1.0 / 255.0
            image_nf = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
            image    = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
            image    = image.unsqueeze(0)
            image_nf = image_nf.unsqueeze(0)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.feed_data(
                data    = {
                    "idx"   : i,
                    "LQs"   : image,
                    "nf"    : image_nf,
                    "border": 0,
                },
                need_GT = False,
            )
            model.test()
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals(need_GT=False)
            enhanced = tensor2img(outputs["rlt"])  # uint8
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
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
