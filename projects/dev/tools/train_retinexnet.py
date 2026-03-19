#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Script.

This script provides a CLI for running RetinexNet training on a given dataset.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root, RunMode, Task
from mon.models.enhance.retinexnet import RetinexNet_Trainer

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    trainer = RetinexNet_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="retinexnet_lol_v1.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="retinexnet",
        model="retinexnet",
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=True,
        verbose=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()

# endregion
