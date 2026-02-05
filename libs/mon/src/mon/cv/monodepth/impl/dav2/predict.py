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
from functools import partial

import box
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
    # 1. Summarize the current run
    if args.verbose:
        mon.print_run_summary(args)

    # 2. Setup environment
    device = mon.create_device(args.device)
    mon.set_random_seed(args.seed)

    # 3. Resolve pre-trained weights
    weights = args.weights or args.resume or args.tuning

    # 4. Define model
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

    # 5. Run benchmark
    if args.benchmark:
        mon.metric.benchmark(model)

    # 6. Resolve I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    '''
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=1),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    '''
    transform  = None
    data_name, dataset = mon.build_dataset(src=args.data, root=args.root, transform=transform)
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

    # 7. Processing loop
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataset),
            total       = len(dataset),
            description = f"[bright_yellow]Predicting"
        ):
            # 7.1 Pre-process
            timers.preprocess.tick()
            meta   = datapoint["meta"]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["imgsz"])
            image  = datapoint["image"]
            timers.preprocess.tock()

            # 7.2. Inference
            timers.infer.tick()
            outputs = model(image, args.imgsz[0])
            timers.infer.tock()

            # 7.3. Post-process
            timers.postprocess.tick()
            depth   = outputs
            depth   = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype("uint8")
            depth   = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            depth_c = mon.depth.to_color(depth)
            timers.postprocess.tock()

            # 7.4. Save
            if args.save:
                # Save to: ".../pred/"
                out_dir  = resolve_output_dir(src_path=path)
                out_path = out_dir / mon.DIRS.DEPTH / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(depth, out_path)

            # 7.5. Save debug
            if args.save_debug:
                # Save to: ".../debug/"
                out_dir  = resolve_debug_dir(src_path=path)
                out_path = out_dir / mon.DIRS.DEPTH / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(depth_c, out_path)
    timers.total.tock()

    # 8. Finish
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
