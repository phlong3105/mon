#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running PairLIE on a given dataset.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

from __future__ import annotations

__all__ = []

import argparse

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.pairlie import PairLIE_Predictor, PairLIE_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.mode == "train":
        trainer = PairLIE_Trainer.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.TRAIN,
            arch="pairlie",
            model="pairlie",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        trainer.train()
    elif args.mode == "predict":
        predictor = PairLIE_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file="pairlie_sice.yaml",
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="pairlie",
            model="pairlie",
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
    parser.add_argument("--config", type=str, default="pairlie_sice.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

# endregion
