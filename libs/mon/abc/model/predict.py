#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""<Model> prediction script.

This script provides a command-line interface for running <model>
prediction on a given dataset.

References:
    - Paper:
    - Code:
"""

from __future__ import annotations

__all__ = []

import copy
import sys
from functools import partial

import box
import cv2
import torch

import mon
from mon import albumentations as A

mon.preload()

current_file = mon.Path(__file__).normalize()
current_dir  = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import dav2' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m dav2.predict
    from .model import Model
except ImportError:
    # Works when running as a script: python predict.py
    from model import Model


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

    # Resolve I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=1),
        A.Normalize(normalization="min_max"),
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
            outputs = model(image, args.imgsz[0])
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            # Already resized in model.infer_image()
            h1, w1  = mon.image.imgsz(outputs)
            if (h1, w1) != (h0, w0):
                outputs = cv2.resize(outputs, (w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save:
                # Save to: ".../pred/"
                out_dir  = resolve_output_dir(src_path=path)
                out_path = out_dir / mon.DIRS.IMAGE / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(outputs, out_path)

            # Save debug
            if args.save_debug:
                # Save to: ".../debug/"
                out_dir  = resolve_debug_dir(src_path=path)
                out_path = out_dir / mon.DIRS.DEPTH / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(outputs, out_path)
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
