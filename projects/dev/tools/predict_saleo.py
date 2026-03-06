#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running SALEO prediction on a given dataset.

References:
    - Paper: "Scale-Arbitrary Low-Light Enhancement via Depth-Aware Implicit
      Neural Optimization"
    - Code: https://github.com/phlong3105/saleo
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from saleo import SALEO_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = SALEO_Predictor.from_cli(
        root=resolve_project_root(current_dir),
        config_file="saleo_ffsiren.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="saleo",
        model="saleo_ffsiren",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    main()

# endregion
