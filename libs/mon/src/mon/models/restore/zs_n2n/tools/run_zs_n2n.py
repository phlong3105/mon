#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running ZS-N2N on a given dataset.

References:
    - Paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
      Data," CVPR 2023.
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA

    - Paper: "Unsupervised Image Denoising Based on Self-Attention Mechanism,"
      ICAICE 2023.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.restore.zs_n2n import ZS_N2N_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.predict or args.train or args.test:
        predictor = ZS_N2N_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file=args.config,
            task=Task.ENHANCE,
            mode=RunMode.PREDICT,
            arch="zs_n2n",
            model="zs_n2n",
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
    parser.add_argument("--config", type=str, default="zs_n2n.yaml")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--predict", action="store_true", help="Predict using the model.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion
