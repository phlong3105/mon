#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PairLIE model prediction pipepline for low-light image enhancement.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

import copy

import box
import cv2
import torch

import mon
import pairlie
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

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
    model = pairlie.PairLIE(weights=pretrained)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
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
            
            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            L, R, X = outputs
            D = image - X
            I = torch.pow(L, 0.2) * R  # default=0.2, LOL=0.14.
            L = mon.image.to_array(L)
            R = mon.image.to_array(R)
            I = mon.image.to_array(I)
            D = mon.image.to_array(D)
            # L_img = transforms.ToPILImage()(L.squeeze(0))
            # R_img = transforms.ToPILImage()(R.squeeze(0))
            # I_img = transforms.ToPILImage()(I.squeeze(0))
            # D_img = transforms.ToPILImage()(D.squeeze(0))
            h1, w1 = mon.image.imgsz(L)
            if (h1, w1) != (h0, w0):
                L = cv2.resize(L,(w0, h0))
                R = cv2.resize(R,(w0, h0))
                I = cv2.resize(I,(w0, h0))
                D = cv2.resize(D,(w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(I, out_path)

            if args.save_debug:
                debug_dir = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                mon.image.save_image(L, debug_dir / f"{path.stem}_L{mon.SAVE_IMAGE_EXT}")
                mon.image.save_image(R, debug_dir / f"{path.stem}_R{mon.SAVE_IMAGE_EXT}")
                mon.image.save_image(D, debug_dir / f"{path.stem}_D{mon.SAVE_IMAGE_EXT}")
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
