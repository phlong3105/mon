#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for enhancement models.
"""

from __future__ import annotations

__all__ = [
    "EnhancementModel",
]

from abc import ABC
from typing import Any, override

import torch

from mon.core import Size, SizeLike
from mon.metrics import benchmark
from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class EnhancementModel(Model, ABC):
    """A base class for all enhancement models."""

    requires: set = {"image"}
    provides: set = {"enhanced"}

    # --- Benchmarks ---
    @override
    def benchmark(self, imgsz: SizeLike, num_runs: int = 10, verbose: bool = True) -> dict[str, Any]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (SizeLike): Input image size.
            num_runs (int, optional): Number of runs to average for benchmarking.
                Defaults to 10.
            verbose (bool, optional): Whether to log the results. Defaults to True.

        Returns:
            dict: A dictionary containing the benchmark results, such as latency,
                FLOPs, and parameter count.
        """
        imgsz = Size.from_value(imgsz)
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = torch.randn(1, 3, imgsz.h, imgsz.w).to(device)
        inputs = {"image": dummy_input}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, num_runs=num_runs, verbose=verbose)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
