#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoLIE prediction script.

This script provides a command-line interface for running CoLIE prediction on a
given dataset.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = []

import copy
import sys
from functools import partial

import box

import mon
from mon.training import albumentations as A

mon.preload()

current_file = mon.Path(__file__).normalize()
current_dir  = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import colie' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m colie.predict
    from .model import colie
except ImportError:
    # Works when running as a script: python predict.py
    from model import colie


# ==============================================================================
# region CONTROL
# ==============================================================================

def run(args: box.Box):
    # 1. Summarize the current run
    if args.verbose:
        mon.print_run_summary(args)

    # 2. Setup environment
    device = mon.create_device(args.device)
    mon.set_random_seed(args.seed)

    # 3. Resolve pre-trained weights
    # Do nothing

    # 4. Define model
    model = colie(device=device, *args.network)
    model = model.to(device)

    # 5. Run benchmark
    if args.benchmark:
        mon.metric.benchmark(model)

    # 6. Resolve I/O
    transform = A.Compose([
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
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
    # Define hyperparameters
    epochs = args.epochs
    E      = args.network.E
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataset),
            total       = len(dataset),
            description = f"[bright_yellow]Predicting"
        ):
            # 7.1. Preprocess
            timers.preprocess.tick()
            meta  = datapoint["meta"]
            path  = mon.Path(meta["path"])
            image = datapoint["image"]
            image = image.to(device)
            timers.preprocess.tock()

            # 7.2. Inference
            timers.infer.tick()
            outputs = model(image, epochs=epochs, E=E, save_debug=args.save_debug)
            timers.infer.tock()

            # 7.3. Post-process
            timers.postprocess.tick()
            enhanced = outputs["enhanced"]
            enhanced = mon.image.to_array(enhanced)
            debug    = {}
            if args.save_debug:
                debug = {
                    "image_i":      mon.image.to_array(outputs["image_i"]),
                    "image_i_res":  mon.image.to_array(outputs["image_i_res"]),
                    "image_i_fixed":mon.image.to_array(outputs["image_i_fixed"]),
                    "image_r":      mon.image.to_array(outputs["image_r"]),
                }
            timers.postprocess.tock()

            # 7.4. Save
            if args.save:
                # Save to: ".../pred/"
                out_dir  = resolve_output_dir(src_path=path)
                out_path = out_dir / f"{path.stem}{mon.EXT.IMAGE}"
                mon.image.write(enhanced, out_path)

            # 7.5. Save debug
            if args.save_debug:
                # Save to: ".../debug/"
                out_dir  = resolve_debug_dir(src_path=path)
                for k, v in debug.items():
                    out_path = out_dir / f"{path.stem}_{k}{mon.EXT.IMAGE}"
                    mon.image.write(v, out_path)
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
