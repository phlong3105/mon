#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction Script.

This script provides a CLI for running Depth Anything V2 prediction on a given
dataset.

References:
    - Paper: "Depth Anything V2. A More Capable Foundation Model for Monocular
      Depth Estimation," NeurIPS 2024.
    - Code: https://github.com/DepthAnything/Depth-Anything-V2
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.monodepth.dav2 import DAV2_Predictor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
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
        exist_ok=True,
        verbose=True,
    )
    predictor.predict()


if __name__ == "__main__":
    main()

# endregion
