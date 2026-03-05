#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running Zero-DCE prediction on a given dataset.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE

    - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation," IEEE TPAMI 2022.
    - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from __future__ import annotations

__all__ = []

import cv2
import torch

from mon import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    metrics,
    MODELS,
    Path,
    RunMode,
    Size,
    Split,
    sys_ctx,
    Task,
    TimeProfiler,
    to_image_array,
    transform as T,
)
from mon.cv import write_image
from mon.dataset import build_dataset

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


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
    weights = config.weights or config.finetune

    # 4. Define model
    imgsz = Size.from_value(config.eval_imgsz)
    scale_factor = config.model.get("scale_factor")
    if scale_factor:
        imgsz = Size(
            height=imgsz.h // scale_factor,
            width=imgsz.w // scale_factor,
        )

    model = MODELS.build(**config.model | { "weights": weights})
    model = model.to(device)
    model.eval()

    # 5. Run benchmark
    if config.benchmark:
        metrics.benchmark(model, imgsz=imgsz)

    # 6. Define transforms
    transforms = T.Compose([
        T.ResizeDivisibleBy(height=imgsz.h, width=imgsz.w, divisor=32),
        T.Normalize(normalization="min_max"),
        T.ToTensorV2(transpose_mask=True),
    ])

    # 7. Prediction loop
    for src in config.data:
        # 7.1. Build dataset
        data_name, dataset = build_dataset(
            src=src,
            dataset_dir=config.data_dir,
            split=Split.TEST,
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
                size0 = Size.from_value(meta["imgsz"])
                image = datapoint["image"]
                image = image.unsqueeze(0).to(device)
                timers.preprocess.tock()

                # 7.2.2. Inference
                timers.infer.tick()
                outputs = model(image, save_debug=config.save_debug)
                timers.infer.tock()

                # 7.2.3. Postprocess
                timers.postprocess.tick()
                enhanced = outputs["enhanced"]
                enhanced = to_image_array(enhanced)
                size1 = Size.from_value(enhanced)
                if size1 != size0:
                    enhanced = cv2.resize(enhanced, size0.wh)
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
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="zero_dce",
        model="zero_dce",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.PREDICT)
    predict(config)


if __name__ == "__main__":
    main()

# endregion
