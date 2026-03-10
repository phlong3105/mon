#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running CALIE prediction on a given dataset.

References:
    - Paper: "Continuously Adjustable Low-Light Implicit Enhancement"
    - Code: https://github.com/phlong3105/calie
"""

from __future__ import annotations

__all__ = []

from calie import CALIE_Predictor
from mon.core import Path, resolve_project_root, RunMode, Task

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = CALIE_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="calie_siren.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="calie",
        model="calie_siren",
        device="cuda:0",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    main()

# endregion
