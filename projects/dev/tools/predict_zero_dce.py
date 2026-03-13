#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running Zero-DCE training on a given dataset.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.zero_dce import ZeroDCE_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = ZeroDCE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="zero_dce_sice_me.yaml",
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


if __name__ == "__main__":
    main()

# endregion
