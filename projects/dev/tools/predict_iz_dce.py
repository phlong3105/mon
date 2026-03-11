#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running IZ-DCE training on a given dataset.

References:
    - Paper: "IZ-DCE: Implicit Zero-Reference Deep Curve Estimation"
    - Code: https://github.com/phlong3105/izdce
"""

from __future__ import annotations

__all__ = []

from iz_dce import IZ_DCE_Predictor
from mon.core import Path, resolve_project_root, RunMode, Task

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = IZ_DCE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="iz_dce_sice_me_v1.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="iz_dce",
        model="iz_dce",
        device="cuda:0",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    main()

# endregion
