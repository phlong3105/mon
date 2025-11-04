#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RetinexNet model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

import copy

import box

import mon
import retinexnet

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    model = retinexnet.RetinexNet(args.imgsz, args.benchmark)
    model = model.to(device)
    
    # Data I/O
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Listing images",
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            timers.preprocess.tock()

            out_dir = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
            # out_dir = out_dir / mon.SAVE_IMAGE_DIR
            out_dir.mkdir(parents=True, exist_ok=True)

            # Infer
            timers.infer.tick()
            model.predict(
                [path],
                res_dir  = str(out_dir),
                ckpt_dir = str(args.weights),
                imgsz    = args.imgsz,
                resize   = args.resize
            )
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
