#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model complexity metrics.

This module provides functions to compute the number of parameters, MACs, and
FLOPs of a given PyTorch model. It also includes a benchmarking function to log
these statistics.
"""

from __future__ import annotations

__all__ = [
    "benchmark",
    "compute_model_stats",
]

import copy
import time

import thop
import torch
from torch import nn

from mon.core import image as I, log


# ==============================================================================
# region MODEL COMPLEXITY METRICS
# ==============================================================================

def compute_model_stats(
    model: nn.Module,
    imgsz: int | tuple[int, int] = 512,
    channels: int = 3,
) -> tuple[float, float, float]:
    """Compute the number of parameters, MACs, and FLOPs of a model.

    Args:
        model (nn.Module): PyTorch model to profile.
        imgsz (int | tuple[int, int]): Input image size. Defaults to 512.
        channels (int): Number of input channels. Defaults to 3.

    Returns:
        tuple[float, float, float]: Number of parameters, MACs, and FLOPs.
    """
    h, w = I.imgsz(imgsz)
    device = next(model.parameters()).device

    # Use a dummy input
    input_data = torch.randn(1, channels, h, w).to(device)

    # Eval mode is crucial for accurate MACs (e.g., skips Dropout)
    model.eval()

    with torch.no_grad():
        # thop.profile often modifies the model with hooks;
        # deepcopy protects the original object
        model_copy = copy.deepcopy(model)
        macs, params = thop.profile(model_copy, inputs=(input_data,), verbose=False)

    flops = 2 * macs
    return params, macs, flops


def benchmark(
    model: nn.Module,
    imgsz: int | tuple[int, int] = 512,
    channels: int = 3,
    num_runs: int = 10,
):
    """Measure and log the complexity of a model.

    Args:
        model (nn.Module): PyTorch model to benchmark.
        imgsz (int | tuple[int, int]): Input image size. Defaults to 512.
        channels (int): Number of input channels. Defaults to 3.
        num_runs (int): Number of runs for latency measurement. Defaults to 10.
    """
    # Compute complexity stats
    params, macs, flops = compute_model_stats(model=model, imgsz=imgsz, channels=channels)

    # Measure Latency (Inference speed)
    h, w = I.imgsz(imgsz)
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, channels, h, w).to(device)
    model.eval()

    # Warmup
    for _ in range(3):
        _ = model(dummy_input)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    avg_latency = (time.perf_counter() - start_time) / num_runs * 1000  # ms

    # Log results with human-readable formatting
    log("-" * 30)
    log(f"{'Model Benchmark':^30}")
    log("-" * 30)
    log(f"Input Shape : ({channels}, {h}, {w})")
    log(f"Params      : {_format_unit(params, 'M')}")
    log(f"MACs        : {_format_unit(macs, 'G')}")
    log(f"FLOPs       : {_format_unit(flops, 'G')}")
    log(f"Latency     : {avg_latency:.2f} ms / image")
    log("-" * 30)
    # log(f"Params    : {params:.4f}")
    # log(f"MACs      : {macs:.4f}")
    # log(f"FLOPs     : {flops:.4f}")


# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _format_unit(val: float, target: str = "M") -> str:
    """Helper to format large numbers (e.g., 1.2G, 3.5M)."""
    if target == "G":
        return f"{val / 1e9:.2f} G"
    return f"{val / 1e6:.2f} M"


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
