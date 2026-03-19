#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running PairLIE prediction on a given dataset.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.pairlie import PairLIE_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
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


if __name__ == "__main__":
    main()

# endregion
