#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running DCC-Net on a given dataset.

References:
    - Paper: "Deep Color Consistent Network for Low Light-Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/Ian0926/DCC-Net
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.dccnet import DCCNet_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.predict:
        predictor = DCCNet_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file="dccnet_lol_v1.yaml",
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="dccnet",
            model="dccnet",
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
    parser.add_argument("--config", type=str, default="dccnet_lol_v1.yaml")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
