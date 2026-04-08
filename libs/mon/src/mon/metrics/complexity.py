#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model Complexity Statistics.

This module provides functions to compute the number of parameters, MACs, and
FLOPs of a given PyTorch model. It also includes a benchmarking function to log
these statistics.
"""

from __future__ import annotations

__all__ = [
    "benchmark",
    "compute_latency",
    "compute_model_stats",
]

import copy
import time
from typing import Any

import thop
import torch
from box import Box
from torch import nn

from mon.core import Float3, log


# ==============================================================================
# region MODEL COMPLEXITY
# ==============================================================================

def compute_model_stats(model: nn.Module, inputs: Any) -> Float3:
    """Compute the number of parameters, MACs, and FLOPs of a model.

    Args:
        model (nn.Module): PyTorch model to profile.
        inputs (Any): Dummy inputs to the model (e.g., a tensor or a dict of tensors).

    Returns:
        Float3: Number of parameters, MACs, and FLOPs.
    """
    # Eval mode is crucial for accurate MACs (e.g., skips Dropout)
    model.eval()

    with torch.no_grad():
        # thop.profile often modifies the model with hooks;
        # deepcopy protects the original object
        model_copy = copy.deepcopy(model)
        macs, params = thop.profile(model_copy, inputs=(inputs,), verbose=False)

    flops = 2 * macs
    return params, macs, flops


def compute_latency(model: nn.Module, inputs: Any, num_runs: int = 10) -> float:
    """Measure the latency of a model.

    Args:
        model (nn.Module): PyTorch model to benchmark.
        inputs (Any): Dummy inputs to the model (e.g., a tensor or a dict of tensors).
        num_runs (int, optional): Number of runs for latency measurement.
            Defaults to 10.

    Returns:
        float: Average latency in milliseconds.
    """
    device = next(model.parameters()).device

    # Eval mode is crucial for accurate MACs (e.g., skips Dropout)
    model.eval()

    # Warmup runs to stabilize memory placement and JIT optimizations
    for _ in range(5):
        _ = model(inputs)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    avg_latency = (time.perf_counter() - start_time) / num_runs * 1000  # ms

    return avg_latency


def benchmark(
    model: nn.Module,
    inputs: Any,
    num_runs: int = 10,
    verbose: bool = True,
) -> dict[str, float]:
    """Measure and log the complexity of a model.

    Args:
        model (nn.Module): PyTorch model to benchmark.
        inputs (Any): Dummy inputs to the model (e.g., a tensor or a dict of tensors).
        num_runs (int, optional): Number of runs for latency measurement.
            Defaults to 10.
        verbose (bool, optional): Whether to log the results. Defaults to True.

    Returns:
        dict[str, float]: A dictionary containing the measured parameters, MACs,
            FLOPs, and latency.
    """
    # Compute complexity stats
    params, macs, flops = compute_model_stats(model=model, inputs=inputs)

    # Compute latency
    latency = compute_latency(model=model, inputs=inputs, num_runs=num_runs)

    # Log results with human-readable formatting
    if verbose:
        if isinstance(inputs, (Box, dict)):
            b, c, h, w = list(inputs.values())[0].shape
        else:
            b, c, h, w = inputs.shape

        log("-" * 30)
        log(f"{'Model Benchmark':^30}")
        log("-" * 30)
        log(f"Input Shape : ({b}, {c}, {h}, {w})")
        log(f"Params      : {_format_unit(params, 'M')}")
        log(f"MACs        : {_format_unit(macs, 'G')}")
        log(f"FLOPs       : {_format_unit(flops, 'G')}")
        log(f"Latency     : {latency:.2f} ms / image")
        log("-" * 30)
        # log(f"Params    : {params:.4f}")
        # log(f"MACs      : {macs:.4f}")
        # log(f"FLOPs     : {flops:.4f}")

    # Return the measured results for further analysis
    return {
        "params": params,
        "macs": macs,
        "flops": flops,
        "latency": latency,
    }

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _format_unit(value: float, target: str = "M") -> str:
    """Helper to format large numbers (e.g., 1.2G, 3.5M)."""
    if target == "G":
        return f"{value / 1e9:.2f} G"
    return f"{value / 1e6:.2f} M"

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
