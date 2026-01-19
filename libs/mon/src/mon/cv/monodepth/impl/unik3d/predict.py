#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UniK3D prediction script.

This script provides a command-line interface for running UniK3D prediction on
 given dataset.

References:
    - Paper: "UniK3D: Universal Camera Monocular 3D Estimation," CVPR 2025.
    - Code: https://github.com/lpiccinelli-eth/UniK3D
"""

from __future__ import annotations

__all__ = []

import copy
import sys
from functools import partial

import box
import numpy as np
import torch

import mon
from mon.training import albumentations as A

mon.preload()

current_file = mon.Path(__file__).normalize()
current_dir  = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import unik3d' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m unik3d.predict
    from .model import UniK3D
except ImportError:
    # Works when running as a script: python predict.py
    from model import UniK3D


# ==============================================================================
# region CONTROL
# ==============================================================================

@torch.no_grad()
def run(args: box.Box):
    # Summarize the current run
    if args.verbose:
        mon.print_run_summary(args)

    # Setup environment
    device = mon.create_device(args.device)
    mon.set_random_seed(args.seed)

    # Resolve pre-trained weights
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

    # Run benchmark
    if args.benchmark:
        mon.metric.benchmark(model)

    # Define I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        # A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=1),
        # A.Normalize(normalization="min_max"),  # Normalization will be taken care of by the model
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataset = mon.build_dataset(args.data, args.root, transform)
    resolve_output_dir = partial(
        mon.resolve_output_dir,
        root         = args.save_dir,
        dirname      = data_name,
        subdir_name  = mon.DIRS.PRED,
        keep_subdirs = args.keep_subdirs,
        save_nearby  = args.save_nearby,
    )
    resolve_debug_dir = partial(
        mon.resolve_output_dir,
        root         = args.save_dir,
        dirname      = data_name,
        subdir_name  = mon.DIRS.DEBUG,
        keep_subdirs = args.keep_subdirs,
        save_nearby  = args.save_nearby,
    )

    # Processing loop
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
            outputs = model(rgb=image, camera=None, normalize=True, rays=None)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            # Metric Depth Estimation
            depth   = outputs["depth"]
            depth   = depth.cpu().numpy().squeeze()
            depth   = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            depth_c = mon.depth.to_color(depth)
            # Point Cloud in Camera Coordinate
            points  = outputs["points"]
            points  = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
            # Unprojected rays
            rays    = outputs["rays"]
            rays    = ((rays + 1) * 127.5).clip(0, 255)
            rays    = rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()
            timers.postprocess.tock()

            # Save
            if args.save:
                # Save to: ".../pred/"
                out_dir  = resolve_output_dir(src_path=path)
                out_path = out_dir / mon.DIRS.DEPTH / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(depth, out_path)

            # Save debug
            if args.save_debug:
                # Save to: ".../debug/"
                out_dir  = resolve_debug_dir(src_path=path)
                out_path = out_dir / mon.DIRS.DEPTH / f"{path.stem}{mon.EXT.IMAGE}"
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
