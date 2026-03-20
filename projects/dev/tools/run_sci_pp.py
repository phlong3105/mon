#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running SCI++ on a given dataset.

References:
    - Paper: "Learning with Self-Calibrator for Fast and Robust Low-Light
      Image Enhancement," TPAMI 2025.
    - Code: https://github.com/vis-opt-group/SCI
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.sci import SCI_PP_Predictor, SCI_PP_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.train:
        trainer = SCI_PP_Trainer.from_cli(
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.TRAIN,
            arch="sci",
            model="sci++",
            device="auto",
            save=True,
            save_debug=True,
            exist_ok=True,
            verbose=True,
        )
        trainer.train()
    elif args.predict:
        predictor = SCI_PP_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="sci",
            model="sci++",
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
    parser.add_argument("--config", type=str, default="sci++.yaml")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
