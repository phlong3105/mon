#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeurOP model prediction pipeline for image retouching.

References:
    - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
    - Code: https://github.com/amberwangyili/neurop
"""

import copy

import box
import cv2
import torch

import mon
from mon import albumentations as A
from neurop import build_model, dict_to_nonedict, parse

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path        = root_dir / "neurop" / "option" / "test" / args.cfg
    cfgs            = parse(str(cfg_path))
    cfgs            = dict_to_nonedict(cfgs)
    cfgs["dist"]    = False
    cfgs["weights"] = args.weights

    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)
    cfgs["device"] = device

    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    model = build_model(cfgs)
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
    
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)
    
    # Predicting
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
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.feed_data(data = {
                "LQ": image,
                "GT": image,
            })
            model.test()
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals()
            enhanced = outputs["rlt"]
            enhanced = (255.0 * enhanced).astype("uint8")
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
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
