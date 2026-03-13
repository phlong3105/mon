#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Script.

This script provides a CLI for running SLICE training on a given dataset.

References:
    - Paper: "SLICE: Scale-Arbitrary Low-Light Enhancement via Depth-Aware
      Implicit Curve Estimation"
    - Code: https://github.com/phlong3105/slice
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from slice import SLICE_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    trainer = SLICE_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="slice_dopri5_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="slice",
        model="slice",
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=False,
        verbose=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()

# endregion
