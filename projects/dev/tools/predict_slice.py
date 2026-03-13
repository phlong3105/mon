#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running SLICE training on a given dataset.

References:
    - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
      Implicit Curve Estimation"
    - Code: https://github.com/phlong3105/slice
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from slice import SLICE_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = SLICE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="slice_sice_me_v2.yaml",
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


if __name__ == "__main__":
    main()

# endregion
