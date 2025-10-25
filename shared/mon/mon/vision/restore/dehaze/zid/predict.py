#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZID model prediction pipeline for image dehazing.

References:
    - Paper: "Zero-Shot Image Dehazing," IEEE TIP 2020.
    - Code: https://github.com/XLearning-SCU/2020-TIP-ZID
"""

import copy

import box

import mon
import zid
from mon import albumentations as A

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
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
            timers.preprocess.tock()

            # Save
            out_dir   = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
            debug_dir = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
            out_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir /    "t").mkdir(parents=True, exist_ok=True)
            (debug_dir /    "a").mkdir(parents=True, exist_ok=True)
            (debug_dir / "mask").mkdir(parents=True, exist_ok=True)
            
            # Infer
            timers.infer.tick()
            dh = zid.ZID(str(path.stem), image, args.epochs, clip=True, output_path=str(out_dir))
            dh.optimize()
            dh.finalize()
            timers.infer.tock()
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
