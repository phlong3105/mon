#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CLODE model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
      Neural ODEs," ICLR 2025.
    - Code: https://github.com/dgjung0220/CLODE
"""

import copy

import box
import cv2
import matplotlib
import torch

import clode
import mon
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
    T     = torch.tensor([0, args.network.T]).float().to(device)
    model = clode.CLODE(weights=pretrained)
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
    cmap   = matplotlib.colormaps.get_cmap("RdBu")
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
            outputs = model(image, T, inference=True)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs["output"]
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            if args.save_debug:
                curve_map = outputs["curve_map"]
                noise_map = outputs["noise_map"]
                ref       = datapoint.get("ref", None)
                image     = mon.image.to_array(image)
                curve_map = mon.image.to_array(curve_map)
                noise_map = mon.image.to_array(noise_map)
                ref       = mon.image.to_array(ref) if ref is not None else None
                # noise_map = noise_map.squeeze().detach().cpu().permute(1, 2, 0).numpy()
                # noise_map = cv2.cvtColor(noise_map, cv2.COLOR_RGB2GRAY)
                # noise_map = (cmap(noise_map)[:, :, :3] * 255).astype("uint8")
                if (h1, w1) != (h0, w0):
                    image     = cv2.resize(image,     (w0, h0))
                    curve_map = cv2.resize(curve_map, (w0, h0))
                    noise_map = cv2.resize(noise_map, (w0, h0))
                    ref       = cv2.resize(ref,       (w0, h0)) if ref is not None else None
                if ref is not None:
                    debug_image = cv2.hconcat([image, enhanced, ref])
                else:
                    debug_image = cv2.hconcat([image, enhanced])
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
            if args.save_debug:
                debug_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                debug_path = debug_dir / f"{path.stem}_curve_map{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(curve_map, debug_path)
                debug_path = debug_dir / f"{path.stem}_noise_map{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(noise_map, debug_path)
                debug_path = debug_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(debug_image, debug_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir, model_root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
