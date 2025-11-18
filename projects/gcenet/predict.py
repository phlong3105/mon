#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCENet model prediction pipeline for low-light image enhancement."""

import copy

import box
import cv2
import torch

# noinspection PyUnusedImports
import gcenet
import mon
from mon import albumentations as A
mon.preload()

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
    args.network |= {
        "name"   : args.model,
        "weights": pretrained,
    }
    model = mon.MODELS.build(**args.network)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.metrics.benchmark(model)
   
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
            # depth  = datapoint.get("depth", None)
            # depth  = depth.to(device) if depth is not None else None
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, inference=True)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs["output"]
            enhanced = mon.image.to_array(enhanced) if isinstance(enhanced, torch.Tensor) else enhanced
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            if args.save_debug:
                image     = mon.image.to_array(image)
                curve_map = mon.image.to_array(outputs["curve_map"])
                noise_map = mon.image.to_array(outputs["noise_map"])
                alls      = [mon.image.to_array(img) for img in outputs["all"]]
                if (h1, w1) != (h0, w0):
                    image     = cv2.resize(image, (w0, h0))
                    curve_map = cv2.resize(curve_map, (w0, h0))
                    noise_map = cv2.resize(noise_map, (w0, h0))
                    alls      = [cv2.resize(img, (w0, h0)) for img in alls]
                debug_image = cv2.hconcat([image, enhanced, curve_map, noise_map])
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}_debug{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(debug_image, out_path)
                for j, img in enumerate(alls):
                    out_path = out_dir / f"{path.stem}_{j}{mon.SAVE_IMAGE_EXT}"
                    mon.image.save_image(img, out_path)
            
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.parse_cli_args(root=root_dir)
    data = mon.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()
