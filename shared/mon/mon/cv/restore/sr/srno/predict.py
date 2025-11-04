#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SRNO model prediction pipeline for super-resolution.

References:
    - Paper: "Super-Resolution Neural Operator," CVPR 2023.
    - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

import copy

import box
import torch

import mon
from mon import albumentations as A
from srno import make, make_coord

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
    model = make(torch.load(pretrained, weights_only=True)["model"], load_sd=True)
    model = model.to(device)
    model.eval()

    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    transform = A.Compose([
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
            meta        = datapoint["meta"][0]
            path        = mon.Path(meta["path"])
            image       = image.to(device)
            h0          = int(image.shape[-2] * int(args.scale))
            w0          = int(image.shape[-1] * int(args.scale))
            scale_      = h0 / image.shape[-2]
            coord       = make_coord((h0, w0), flatten=False).to(device)
            cell        = torch.ones(1, 2).to(device)
            cell[:, 0] *= 2 / h0
            cell[:, 1] *= 2 / w0
            cell_factor = max(scale_ / args.scale_max, 1)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(
                inp   = ((image - 0.5) / 0.5).to(device),
                coord = coord.unsqueeze(0),
                cell  = cell_factor * cell
            )
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = (outputs * 0.5 + 0.5)
            enhanced = mon.image.to_array(enhanced)
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
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
