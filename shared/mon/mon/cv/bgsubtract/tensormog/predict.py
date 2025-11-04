#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements TensorMoG model prediction pipeline for background subtraction.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic Scene
      Adaptation for Background Modeling," Sensors 2020.
"""

import copy

import box
import cv2

import mon
import tensormog
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
# @torch.no_grad()
def predict(args: dict | box.Box) -> str:
    height            = args.network.height
    width             = args.network.width
    num_gaussians     = args.network.num_gaussians
    learning_rate     = args.network.learning_rate
    matching_thres    = args.network.matching_thres
    background_thres  = args.network.background_thres
    num_updates       = args.network.num_updates
    tau_rate          = args.network.tau_rate
    tau_updating_rate = args.network.tau_updating_rate
    
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    model = tensormog.TensorMOG(
        height            = height,
        width             = width,
        num_gaussians     = num_gaussians,
        learning_rate     = learning_rate,
        matching_thres    = matching_thres,
        background_thres  = background_thres,
        num_updates       = num_updates,
        tau_rate          = tau_rate,
        tau_updating_rate = tau_updating_rate,
    )
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model.model)
    
    # Data I/O
    transform = A.Compose([
        A.ResizeDivisibleBy(height=height, width=width, divisor=32),
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
            
            # Optimize and Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            foreground = outputs[0]
            background = outputs[1]
            if h0 != height or w0 != width:
                foreground = cv2.resize(foreground, (w0, h0))
                background = cv2.resize(background, (w0, h0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(background, out_path)
            
            # Save Debug
            if args.save_debug:
                debug_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                debug_path = debug_dir / f"{path.stem}_foreground{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(foreground, debug_path)
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
