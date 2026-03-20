#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running Zero-DCE on a given dataset.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.zero_dce import ZeroDCE_Predictor, ZeroDCE_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.train:
        trainer = ZeroDCE_Trainer.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.TRAIN,
            arch="zero_dce",
            model="zero_dce",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        trainer.train()
    elif args.predict:
        predictor = ZeroDCE_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="zero_dce",
            model="zero_dce",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        predictor.predict()
    else:
        raise NotImplementedError("Run mode hasn't been implemented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--config", type=str, default="zero_dce_sice_me.yaml")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
