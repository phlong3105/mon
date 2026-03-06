#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running CALIE prediction on a given dataset.

References:
    - Paper: "Continuously Adjustable Low-Light Implicit Enhancement"
    - Code: https://github.com/phlong3105/calie
"""

from __future__ import annotations

__all__ = []

import calie
from mon import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    metrics,
    Path,
    resolve_project_root,
    RunMode,
    Split,
    sys_ctx,
    TimeProfiler,
    to_image_array,
    transform as T,
)
from mon.dataset import build_dataset
from mon.ops import write_image

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

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
    model = calie(device=device, **config.model)
    model = model.to(device)

    # 5. Run benchmark
    if config.benchmark:
        metrics.benchmark(model)

    # 6. Define transforms
    transforms = T.Compose([
        T.Normalize(normalization="min_max"),
        T.ToTensorV2(transpose_mask=True),
    ])

    # 7. Prediction loop
    epochs = config.epochs
    E = config.loss.E

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
                image = datapoint["image"]
                depth = datapoint["depth"]
                image = image.unsqueeze(0).to(device)
                depth = depth.unsqueeze(0).to(device) if depth is not None else None
                timers.preprocess.tock()

                # 7.2.2. Inference
                timers.infer.tick()
                outputs = model(
                    image=image,
                    epochs=epochs,
                    E=E,
                    color_func="hvi",
                    save_debug=config.save_debug
                )
                timers.infer.tock()

                # 7.2.3. Postprocess
                timers.postprocess.tick()
                enhanced = outputs["enhanced"]
                enhanced = to_image_array(enhanced)
                debug = {}
                if config.save_debug:
                    debug = {
                        "image_i": to_image_array(outputs["image_i"]),
                        "image_i_res": to_image_array(outputs["image_i_res"]),
                        "image_i_fixed": to_image_array(outputs["image_i_fixed"]),
                        "image_r": to_image_array(outputs["image_r"]),
                    }
                timers.postprocess.tock()

                # 7.2.4. Save
                if config.save:
                    # Save to: ".../pred/"
                    save_path = config.resolve_save_file(K.PRED_DIR, src_path=path)
                    # save_path = save_dir / f"{path.stem}{K.IMAGE_EXT}"
                    write_image(enhanced, save_path)

                # 7.2.5. Save debug
                if config.save_debug:
                    # Save to: ".../debug/"
                    save_dir = config.resolve_save_dir(K.DEBUG_DIR, src_path=path)
                    for k, v in debug.items():
                        save_path = save_dir / f"{path.stem}_{k}{K.IMAGE_EXT}"
                        write_image(v, save_path)
        timers.total.tock()

        # 7.3. Finish
        timers.print()

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    # Load config
    root = resolve_project_root(current_dir)
    config_ctx = ConfigContext.from_cli(
        root=root,
        config_file="calie_siren.yaml",
        task="enhance",
        mode="predict",
        arch="calie",
        model="calie_siren",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.PREDICT)
    predict(config)


if __name__ == "__main__":
    main()

# endregion
