#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for restoration models.
"""

from __future__ import annotations

__all__ = [
    "RestorationModel",
]

from abc import ABC
from typing import override

from mon.core import Size, SizeLike
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class RestorationModel(Model, ABC):
    """A base class for all restoration models."""

    requires: set = {"image"}
    provides: set = {"restored"}

    # --- Benchmarks ---
    @override
    def benchmark(self, imgsz: SizeLike, num_runs: int = 10, verbose: bool = True) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (SizeLike): Input image size.
            num_runs (int, optional): Number of runs to average for benchmarking.
                Defaults to 10.
            verbose (bool, optional): Whether to log the results. Defaults to True.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        imgsz = Size.from_value(imgsz)
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = {"image": dummy_input}
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, num_runs=num_runs, verbose=verbose)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
