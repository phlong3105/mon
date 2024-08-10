#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/pvnieo/Low-light-Image-Enhancement

from __future__ import annotations

import argparse

import cv2

import mon
from exposure_enhancement import enhance_image_exposure

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = mon.Path(args.save_dir)
    weights   = args.weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            image      = datapoint.get("input")
            meta       = datapoint.get("meta")
            image_path = meta["path"]
            h, w       = mon.get_image_size(image)
            if resize:
                image = cv2.resize(image, (imgsz, imgsz))
            timer.tick()
            enhanced_image = enhance_image_exposure(
                im      = image,
                gamma   = args.gamma,
                lambda_ = args.lambda_,
                dual    = not args.lime,
                sigma   = args.sigma,
                bc      = args.bc,
                bs      = args.bs,
                be      = args.be,
                eps     = args.eps
            )
            timer.tock()
            if resize:
                enhanced_image = cv2.resize(enhanced_image, (w, h))
            output_path = save_dir / image_path.name
            cv2.imwrite(str(output_path), enhanced_image)
    avg_time = float(timer.avg_time)
    console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    args.gamma   = 0.6   # Gamma correction parameter
    args.lambda_ = 0.15  # The weight for balancing the two terms in the illumination refinement optimization objective
    args.lime    = True  # Use the LIME method. By default, the DUAL method is used
    args.sigma   = 3     # Spatial standard deviation for spatial affinity based Gaussian weights
    args.bc      = 1     # Parameter for controlling the influence of Mertens's contrast measure
    args.bs      = 1     # Parameter for controlling the influence of Mertens's saturation measure
    args.be      = 1     # Parameter for controlling the influence of Mertens's well exposedness measure
    args.eps     = 1e-3  # Constant to avoid computation instability
    predict(args)


if __name__ == "__main__":
    main()
# endregion
