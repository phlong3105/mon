#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running CoLIE prediction on a given dataset.

References:
    - Paper: "Fast Context-Based Low-Light Image Enhancement via Neural Implicit
      Representations," ECCV 2024.
    - Code: https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.colie import CoLIE_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    predictor = CoLIE_Predictor.from_cli(
        root=resolve_project_root(current_dir),
        config_file="colie.yaml",
        task=Task.ENHANCE,
        mode=RunMode.PREDICT,
        arch="colie",
        model="colie",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    main()

# endregion
