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

import sys

import cv2
import torch

from mon import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    metrics,
    parse_imgsz,
    Path,
    sys_ctx,
    TimeProfiler,
    to_image_array,
    transform as T,
)
from mon.cv import write_image
from mon.dataset import build_dataset

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
def predict(config: Config):
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
        metrics.benchmark(model)

    # 6. Define transforms
    imgsz = parse_imgsz(config.eval_imgsz)
    transforms = T.Compose([
        T.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        T.Normalize(normalization="min_max"),
        T.ToTensorV2(transpose_mask=True),
    ])

    # 7. Process data
    for src in config.data:
        # 7.1. Build dataset
        data_name, dataset = build_dataset(
            src=src,
            dataset_dir=config.data_dir,
            transforms=transforms,
        )

        # 7.2. Main processing loop
        timers = TimeProfiler()
        timers.total.tick()
        with create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence=enumerate(dataset),
                total=len(dataset),
                description=f"[bright_yellow]Predicting"
            ):
                # 7.2.1. Preprocess
                timers.preprocess.tick()
                meta = datapoint["meta"]
                path = Path(meta["path"])
                h0, w0 = parse_imgsz(meta["imgsz"])
                image = datapoint["image"]
                image = image.to(device)
                timers.preprocess.tock()

                # 7.2.2. Inference
                timers.infer.tick()
                outputs = model(image, save_debug=config.save_debug)
                timers.infer.tock()

                # 7.2.3. Postprocess
                timers.postprocess.tick()
                enhanced = outputs["enhanced"]
                enhanced = to_image_array(enhanced)
                h1, w1 = parse_imgsz(enhanced)
                if (h1, w1) != (h0, w0):
                    enhanced = cv2.resize(enhanced, (w0, h0))
                timers.postprocess.tock()

                # 7.2.4. Save
                if config.save:
                    # Save to: ".../pred/"
                    save_path = config.resolve_save_file(K.PRED_DIR, src_path=path)
                    # save_path = save_dir / f"{path.stem}{K.IMAGE_EXT}"
                    write_image(enhanced, save_path)

                # 7.2.5. Save debug
                # Do nothing
        timers.total.tock()

        # 7.3. Finish
        timers.print()

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    # Load config
    config_ctx = ConfigContext.from_cli(
        root=current_dir,
        config_file="zero_dce_sice_me.yaml",
    )
    config = config_ctx.config_for("predict")
    predict(config)


if __name__ == "__main__":
    main()

# endregion
