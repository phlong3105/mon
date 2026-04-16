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

from tensordict import TensorDict

from mon.core import Size, SizeLike
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class RestorationModel(Model, ABC):
    """A base class for all restoration models."""

    in_keys: set = {"image"}
    out_keys: set = {"restored"}

    # --- Benchmarks ---
    @override
    def benchmark(self, imgsz: SizeLike, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (SizeLike): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        imgsz = Size.from_value(imgsz)
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=[])
        inputs = {"data": data}

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
