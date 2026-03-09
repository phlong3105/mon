#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Script.

This script provides a CLI for running IZ-DCE training on a given dataset.

References:
    - Paper: "IZ-DCE: Implicit Zero-Reference Deep Curve Estimation"
    - Code: https://github.com/phlong3105/izdce
"""

from __future__ import annotations

__all__ = []

from iz_dce import IZDCE_ODE_Trainer
from mon.core import Path, resolve_project_root, RunMode, Task

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    trainer = IZDCE_ODE_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="iz_dce_ode_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="iz_dce",
        model="iz_dce_ode",
        device="cuda:1",
        save=True,
        save_debug=True,
        exist_ok=False,
        verbose=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()

# endregion
