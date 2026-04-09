#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metric Evaluation Script.

This script provides a simple interface for measuring IQA metrics.
"""

from __future__ import annotations

__all__ = []

from mon.core import Path, resolve_project_root
from mon.runners import IQAEvaluator

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

def measure():
    # 1. Define arguments
    archs_models = {
        "clode": ["clode_sice_me", "clode_sice_me_gf"],
        "colie": ["colie"],
        "pairlie": ["pairlie_sice", "pairlie_sice_gf"],
        "retinexnet": ["retinexnet_lol_v1", "retinexnet_lol_v1_gf"],
        "sci": ["sci++"],
        "zero_dce": ["zero_dce_sice_me"],
        "zero_ig": ["zero_ig_lol", "zero_ig_lol_gf"],
        "slice": ["slice_dopri5_sice_me_v4"],
    }
    datasets = [
        # "dicm", "lime", "mef", "npe", "vv",
        # "lol_v1",
        # "lol_v2_real",
        # "lol_v2_syn",
        "sice",
        # "lsrw",
        # "uhd_ll",
    ]
    metrics = ["psnr", "ssim", "ssimc", "lpips", "niqe", "pi"]

    # 2. Define constants
    root = resolve_project_root(current_dir)
    if root is None:
        raise FileNotFoundError(f"Could not find project root from {current_dir}")

    data_dir = root / "data"
    run_dir = root / "run" / "predict"

    target_dirs = {
        "lol_v2_real": data_dir / "lol_v2/real/test/target",
        "lol_v2_syn": data_dir / "lol_v2/syn/test/target",
        "sice": data_dir / "sice/sice/test/target",
    }

    # 3. Main loop
    for data in datasets:
        # 3.1. Define the target directory
        if data in target_dirs:
            target_dir = target_dirs[data]
        else:
            target_dir = data_dir / data / "test" / "target"
        if not target_dir.exists():
            target_dir = None

        # 3.2. Loop through the architectures and models
        for arch, models in archs_models.items():
            for model in models:
                input_dir = run_dir / arch / model / data / "pred"
                iqa = IQAEvaluator.from_cli(
                    input_dir=input_dir,
                    target_dir=target_dir,
                    result_file=None,
                    arch=arch,
                    model=model,
                    data=data,
                    metrics=metrics,
                    device="cuda:0",
                    resize=False,
                    verbose=True,
                )
                iqa.measure()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    measure()

# endregion
