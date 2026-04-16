#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running a model on a given dataset.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

# noinspection PyUnusedImports
import slice
from mon.core import (
    ConfigContext,
    Path,
    PREDICTORS,
    resolve_project_root,
    RunMode,
    Task,
    TRAINERS,
)
from mon.runners import Benchmarker, IQAEvaluator

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region FUNCTIONS
# ==============================================================================

def train(args: argparse.Namespace):
    config_ctx = ConfigContext.from_cli(
        root=resolve_project_root(current_dir),
        config_file=args.config,
        task=Task.LLE,
        mode=RunMode.TRAIN,
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.TRAIN, prompt=args.prompt)
    trainer = TRAINERS.build(name=config.model_name, config=config)
    trainer.train()


def predict(args: argparse.Namespace):
    config_ctx = ConfigContext.from_cli(
        root=resolve_project_root(current_dir),
        config_file=args.config,
        task=Task.LLE,
        mode=RunMode.PREDICT,
        device="auto",
        save=True,
        save_debug=False,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.PREDICT, prompt=args.prompt)
    predictor = PREDICTORS.build(name=config.model_name, config=config)
    predictor.predict()


def metric(args: argparse.Namespace):
    # 1. Define arguments
    archs_models = {
        "clode": [
            "clode_sice_me_128",
            "clode_sice_me_256",
            "clode_sice_me_512",
            "clode_sice_me_480p",
            "clode_sice_me_720p",
            "clode_sice_me_1080p",
            "clode_sice_me_2k",
        ],
        # "colie": ["colie"],
        # "pairlie": ["pairlie_sice"],
        # "retinexnet": ["retinexnet_lol_v1"],
        # "sci": ["sci++"],
        "zero_dce": [
            "zero_dce_sice_me_128",
            "zero_dce_sice_me_256",
            "zero_dce_sice_me_512",
            "zero_dce_sice_me_480p",
            "zero_dce_sice_me_720p",
            "zero_dce_sice_me_1080p",
            "zero_dce_sice_me_2k",
            "zero_dce_sice_me_4k",
        ],
        # "zero_ig": ["zero_ig_lol"],
        "slice": [
            "slice_dopri5_sice_me_v4",
        ],
    }
    datasets = [
        # "dicm", "lime", "mef", "npe", "vv",
        # "lol_v1",
        # "lol_v2_real",
        # "lol_v2_syn",
        "sice",
        # "lsrw",
        # "uhd_ll",
    ]
    metrics = [
        # "psnr", "ssim", "ssimc", "lpips", "niqe", "pi",
        "ciqs",
    ]

    # 2. Define constants
    root = resolve_project_root(current_dir)
    if root is None:
        raise FileNotFoundError(f"Could not find project root from {current_dir}")

    data_dir = root / "data"
    run_dir = root / "run" / "predict"

    target_dirs = {
        "lol_v2_real": data_dir / "lol_v2/real/test/target",
        "lol_v2_syn": data_dir / "lol_v2/syn/test/target",
        "sice": data_dir / "sice/sice/test/target",
    }

    # 3. Measure metrics
    for data in datasets:
        # 3.1. Define the target directory
        if data in target_dirs:
            target_dir = target_dirs[data]
        else:
            target_dir = data_dir / data / "test" / "target"
        if not target_dir.exists():
            target_dir = None

        # 3.2. Loop through the architectures and models
        for arch, models in archs_models.items():
            for model in models:
                input_dir = run_dir / arch / model / data / "pred"
                iqa = IQAEvaluator.from_cli(
                    input_dir=input_dir,
                    target_dir=target_dir,
                    result_file=None,
                    arch=arch,
                    model=model,
                    data=data,
                    metrics=metrics,
                    device="cuda:0",
                    resize=False,
                    verbose=True,
                )
                iqa.measure()


def benchmark(args: argparse.Namespace):
    models = [
        "clode",
        # "colie",
        # "pairlie",
        # "retinexnet",
        # "sci++",
        # "zero_dce",
        # "zero_ig",
        # "slice",
    ]
    resolutions = [
        (128, 128),
        (256, 256),
        (512, 512),
        (640, 480),    # 480p
        (1280, 720),   # 720p
        (1920, 1080),  # 1080p
        (2560, 1440),  # 2K
        (3840, 2160),  # 4K
        (7680, 4320),  # 8K
    ]
    for res in resolutions:
        benchmarker = Benchmarker.from_cli(
            task=Task.LLE,
            models=models,
            imgsz=res,
            num_runs=10,
            device="cuda:0",
            verbose=True,
            prompt=args.prompt,
        )
        benchmarker.measure()

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    """A hub for running models."""
    # Train
    if args.train:
        train(args)
    # Predict
    elif args.predict:
        predict(args)
    # Metric
    elif args.metric:
        metric(args)
    # Benchmark
    elif args.benchmark:
        benchmark(args)
    else:
        raise NotImplementedError("Run mode hasn't been implemented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    parser.add_argument("--metric", action="store_true", help="Evaluate the model.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the model.")
    parser.add_argument("--prompt", action="store_true", help="Prompt for additional inputs.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
