#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a command-line interface for running UniK3D prediction on
a given dataset.

References:
    - Paper: "UniK3D: Universal Camera Monocular 3D Estimation," CVPR 2025.
    - Code: https://github.com/lpiccinelli-eth/UniK3D
"""

from __future__ import annotations

__all__ = []

import matplotlib
import numpy as np
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
    transform as T,
)
from mon.dataset import build_dataset
from mon.ops import write_image

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
    # weights = config.weights or config.finetune
    weights = "default"

    # 4. Define model
    imgsz = Size.from_value(config.imgsz)

    model = MODELS.build(device=device, **config.model | { "weights": weights})
    model = model.to(device)
    model.eval()

    # 5. Run benchmark
    if config.benchmark:
        metrics.benchmark(model, imgsz=imgsz)

    # 6. Define transforms
    transforms = T.Compose([
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
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")

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
                image = datapoint["image"]
                timers.preprocess.tock()

                # 7.2.2. Inference
                timers.infer.tick()
                outputs = model(rgb=image, camera=None, normalize=True, rays=None)
                timers.infer.tock()

                # 7.2.3. Postprocess
                timers.postprocess.tick()
                # Metric depth estimation
                depth = outputs["depth"]
                depth = depth.cpu().numpy().squeeze()
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                depth_c = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                # Point cloud in camera coordinate
                points = outputs["points"]
                points = points.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
                # Unprojected rays
                rays = outputs["rays"]
                rays = ((rays + 1) * 127.5).clip(0, 255)
                rays = rays.squeeze().permute(1, 2, 0).byte().cpu().numpy()
                timers.postprocess.tock()

                # 7.2.4. Save
                if config.save:
                    # Save to: ".../pred/"
                    save_path = config.resolve_save_file(K.DEPTH_DIR, src_path=path)
                    # save_path = save_dir / f"{path.stem}{K.IMAGE_EXT}"
                    write_image(depth, save_path)

                # 7.2.5. Save debug
                if config.save_debug:
                    # Save to: ".../debug/"
                    save_path = config.resolve_save_file(K.DEBUG_DIR, src_path=path)
                    # save_path = save_dir / f"{path.stem}{K.IMAGE_EXT}"
                    write_image(depth_c, save_path)
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
        config_file="unik3d_vitb.yaml",
        task=Task.MONODEPTH,
        mode=RunMode.PREDICT,
        arch="unik3d",
        model="unik3d_vitb",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.PREDICT)
    predict(config)


if __name__ == "__main__":
    main()

# endregion
