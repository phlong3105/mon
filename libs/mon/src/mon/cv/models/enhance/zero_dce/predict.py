#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running Zero-DCE prediction on a given dataset.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

from __future__ import annotations

__all__ = []

import copy
import sys
from functools import partial

import cv2
import torch
from box import Box

import mon
from mon import sys_ctx, Path, Config
from mon.dataset import transform as T

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import zero_dce' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zero_dce.predict
    from .model import zero_dce
except ImportError:
    # Works when running as a script: python predict.py
    from model import zero_dce


# ==============================================================================
# region CONTROL
# ==============================================================================

@torch.no_grad()
def run(config: Config):
    # 1. Summarize the current run
    if config.verbose:
        config.log_summary()

    # 2. Setup environment
    device = config.device
    sys_ctx.set_random_seed(config.seed)

    # 3. Resolve pre-trained weights
    # weights = config.weights or config.finetune

    # 4. Define model
    model = zero_dce(**config.model)
    model = model.to(device)
    model.eval()

    # 5. Run benchmark
    if config.benchmark:
        mon.metrics.benchmark(model)

    # 6. Resolve I/O
    imgsz = config.imgsz if config.resize else (0, 0)
    transform = T.Compose([
        T.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        T.Normalize(normalization="min_max"),
        T.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataset = mon.build_dataset(src=config.data, root=config.root, transform=transform)
    resolve_output_dir = partial(
        mon.resolve_output_dir,
        root=config.save_dir,
        dirname=data_name,
        subdir_name=mon.DIRS.PRED,
        keep_subdirs=config.keep_subdirs,
        save_nearby=config.save_nearby,
    )
    resolve_debug_dir = partial(
        mon.resolve_output_dir,
        root=config.save_dir,
        dirname=data_name,
        subdir_name=mon.DIRS.DEBUG,
        keep_subdirs=config.keep_subdirs,
        save_nearby=config.save_nearby,
    )

    # 7. Processing loop
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence=enumerate(dataset),
            total=len(dataset),
            description=f"[bright_yellow]Predicting"
        ):
            # 7.1. Preprocess
            timers.preprocess.tick()
            meta = datapoint["meta"]
            path = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["imgsz"])
            image = datapoint["image"]
            image = image.to(device)
            timers.preprocess.tock()

            # 7.2. Inference
            timers.infer.tick()
            outputs = model(image, save_debug=config.save_debug)
            timers.infer.tock()

            # 7.3. Post-process
            timers.postprocess.tick()
            enhanced = outputs["enhanced"]
            enhanced = mon.image.to_array(enhanced)
            h1, w1 = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()

            # 7.4. Save
            if config.save:
                # Save to: ".../pred/"
                out_dir = resolve_output_dir(src_path=path)
                out_path = out_dir / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(enhanced, out_path)

            # 7.5. Save debug
            # Do nothing
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
