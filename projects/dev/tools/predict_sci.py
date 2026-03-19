#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running SCI prediction on a given dataset.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.sci import SCI_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = SCI_Predictor.from_cli(
        prompt=True,
        root=resolve_project_root(current_dir),
        config_file="sci_medium.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="sci",
        model="sci",
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
