#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Running Script.

This script provides a CLI for running Depth Anything V2 on a given dataset.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = []

import argparse

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.monodepth.dav2 import DAV2_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    if args.mode == "predict":
        predictor = DAV2_Predictor.from_cli(
            prompt=True,
            root=resolve_project_root(current_dir),
            config_file="dav2_vitb_da2k.yaml",
            task=Task.MONODEPTH,
            mode=RunMode.PREDICT,
            arch="dav2",
            model="dav2_vitb",
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
    parser.add_argument("--config", type=str, default="dav2_vitb_da2k.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

# endregion
