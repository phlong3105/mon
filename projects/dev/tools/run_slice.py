#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running SLICE on a given dataset.

References:
    - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
      Implicit Curve Estimation"
    - Code: https://github.com/phlong3105/slice
"""

from __future__ import annotations

__all__ = []

import argparse

from mon.core import Path, resolve_project_root, RunMode, Task
from slice import SLICE_Predictor, SLICE_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.mode == "train":
        trainer = SLICE_Trainer.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.TRAIN,
            arch="slice",
            model="slice",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        trainer.train()
    elif args.mode == "predict":
        predictor = SLICE_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="slice",
            model="slice",
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
    parser.add_argument("--config", type=str, default="slice_sice_me_v1.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

# endregion
