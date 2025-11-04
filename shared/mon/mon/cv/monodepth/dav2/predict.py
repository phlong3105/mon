#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DAV2 model prediction pipeline for monocular depth estimation.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

import copy

import box
import matplotlib
import numpy as np
import torch

import dav2
import mon

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
    model = dav2.DAV2(
        encoder      = args.network.encoder,
        features     = args.network.features,
        out_channels = args.network.out_channels,
        device       = device
    )
    model.load_state_dict(torch.load(str(pretrained), map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    # transform = A.Compose([
        # A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=1),
        # A.Normalize(normalization="min_max"),
        # A.ToTensorV2(transpose_mask=True),
    # ])
    transform = None
    data_name, dataset = mon.data.build_dataset(args.data, args.root, transform)

    # Predict
    cmap   = matplotlib.colormaps.get_cmap("Spectral_r")
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataset),
            total       = len(dataset),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.infer_image(image, args.imgsz[0])
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            # Already resized in model.infer_image()
            # h1, w1  = mon.image.imgsz(outputs)
            # if (h1, w1) != (h0, w0):
            #     outputs = cv2.resize(outputs, (w0, h0))
            depth   = outputs
            depth   = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype("uint8")
            depth   = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            depth_c = (cmap(outputs)[:, :, :3] * 255).astype("uint8")
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(depth, out_path)

            if args.save_debug:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                if args.save_nearby:
                    out_dir = out_dir.parent / f"{out_dir.stem}_c"
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(depth_c, out_path)
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
