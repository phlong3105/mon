#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running Zero-IG on a given dataset.

References:
    - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
      Enhancement for Low-Light Images," CVPR 2024.
    - Code: https://github.com/Doyle59217/ZeroIG
"""

from __future__ import annotations

__all__ = []

import argparse

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.zero_ig import ZeroIG_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.mode == "predict":
        predictor = ZeroIG_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="zero_ig",
            model="zero_ig",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        predictor.predict()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "predict"])
    parser.add_argument("--config", type=str, default="zero_ig_lol.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())


# endregion
