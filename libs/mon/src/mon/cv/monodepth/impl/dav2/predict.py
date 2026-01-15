#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth Anything V2 prediction script.

This script provides a command-line interface for running Depth Anything V2
prediction on a given dataset.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = []

import copy
import sys

import box
import matplotlib
import numpy as np
import torch

import mon

mon.preload()

current_file = mon.Path(__file__).normalize()
current_dir  = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import dav2' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m dav2.predict
    from .model import DAV2
except ImportError:
    # Works when running as a script: python predict.py
    from model import DAV2


# ==============================================================================
# region CONTROL
# ==============================================================================

@torch.no_grad()
def run(args: box.Box):
    # Log a summary of the run
    if args.verbose:
        mon.print_run_summary(args)

    # Setup environment
    device = mon.create_device(args.device)
    mon.set_random_seed(args.seed)

    # Prepare pre-trained weights
    weights = args.weights or args.resume or args.tuning

    # Define model
    model = mon.MODELS.build(
        name    = args.model,
        arch    = args.arch,
        weights = weights,
        device  = device,
        verbose = args.verbose,
        **args.network,
    )
    model = model.to(device)
    model.eval()

    # Run benchmark if specified
    if args.benchmark:
        mon.metric.benchmark(model)

    # Define data loader
    imgsz     = args.imgsz if args.resize else (0, 0)
    '''
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=1),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    '''
    transform = None
    data_name, dataset = mon.build_dataset(args.data, args.root, transform)

    # Processing loop
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
            h0, w0 = mon.image.imgsz(meta["imgsz"])
            image  = datapoint["image"]
            timers.preprocess.tock()

            # Inference
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
                out_dir  = mon.resolve_output_dir(args.save_dir, data_name, mon.DIRS.IMAGE, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(depth, out_path)

            if args.save_debug:
                out_dir  = mon.resolve_output_dir(args.save_dir, data_name, mon.DIRS.DEBUG, path, args.keep_subdirs, args.save_nearby)
                if args.save_nearby:
                    out_dir = out_dir.parent / f"{out_dir.stem}_c"
                out_path = out_dir / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(depth_c, out_path)
    timers.total.tock()

    # Finish
    timers.print()

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    # Parse CLI arguments
    cli  = mon.parse_cli_args(root=current_file)
    data = mon.to_list(cli.data)

    # Run prediction for each dataset
    for d in data:
        cli_      = copy.deepcopy(cli)
        cli_.data = d
        args_     = mon.parse_predict_args(
            cli        = cli_,
            root       = current_dir,
            model_root = current_dir,
        )
        run(args_)


if __name__ == "__main__":
    main()

# endregion
