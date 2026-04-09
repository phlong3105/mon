#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark Script.

This script provides a simple interface for benchmarking models.
"""

from __future__ import annotations

__all__ = []

from mon.core import Path
from mon.runners import BenchmarkEvaluator

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    models = [
        "clode",
        "colie",
        "pairlie",
        "retinexnet",
        "sci++",
        "zero_dce",
        "zero_ig",
        "slice",
    ]
    benchmark = BenchmarkEvaluator.from_cli(models=models, device="cuda:0")
    benchmark.measure()


if __name__ == "__main__":
    main()

# endregion
