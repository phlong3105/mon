#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DarkIR model prediction pipeline for low-light deblurring.

References:
    - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
    - Code: https://github.com/cidautai/DarkIR
"""

import copy

import box
import cv2
import torch
from ptflops import get_model_complexity_info

import mon
from mon import albumentations as A
import darkir

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def load_model(model, path_weights):
    checkpoints = torch.load(str(path_weights), map_location="cpu", weights_only=False)
    weights     = checkpoints["params"]
    # weights     = {"module." + key: value for key, value in weights.items()}
    model.load_state_dict(weights)
    # print("Loaded weights correctly")
    return model


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
    model, _, _ = darkir.create_model(args["network"], rank=0, device=device, torchrun=args.torchrun)
    model = load_model(model, path_weights=pretrained)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    if imgsz[0] >= 1500 or imgsz[1] >= 1500:
        imgsz = (imgsz[0] // 2, imgsz[1] // 2)
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
            outputs = model(image, side_loss=False)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h0, w0) != (h1, w1):
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
