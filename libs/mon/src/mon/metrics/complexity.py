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
    "create_dummy_image",
]

import time
from copy import deepcopy
from typing import Any

import cv2
import thop
import torch
from box import Box
from torch import nn, Tensor

from mon.core import Float3, K, log, Size, SizeLike
from mon.ops import read_image, to_image_tensor


# ==============================================================================
# region MODEL COMPLEXITY
# ==============================================================================

def compute_model_stats(model: nn.Module, inputs: Any, copy: bool = True) -> Float3:
    """Compute the number of parameters, MACs, and FLOPs of a model.

    Args:
        model (nn.Module): PyTorch model to profile.
        inputs (Any): Dummy inputs to the model (e.g., a tensor or a dict of tensors).
        copy (bool, optional): Whether to make a copy of the model before profiling.
            Defaults to True.

    Returns:
        Float3: Number of parameters, MACs, and FLOPs.
    """
    # Normalize inputs
    if isinstance(inputs, (Box, dict)):
        inputs = tuple(inputs.values())
    inputs = (inputs, ) if not isinstance(inputs, tuple) else inputs

    # Eval mode is crucial for accurate MACs (e.g., skips Dropout)
    model.eval()  # NO NEED, some models perform online learning

    with torch.no_grad():
        # thop.profile often modifies the model with hooks;
        # deepcopy protects the original object
        if copy:
            profile_model = deepcopy(model)
        else:
            profile_model = model
        macs, params = thop.profile(profile_model, inputs=inputs, verbose=False)

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
    # Normalize inputs
    device = next(model.parameters()).device

    # Eval mode is crucial for accurate MACs (e.g., skips Dropout)
    model.eval()  # NO NEED, some models perform online learning

    # Warmup runs to stabilize memory placement and JIT optimizations
    for _ in range(5):
        _ = model(**inputs)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    avg_latency = (time.perf_counter() - start_time) / num_runs * 1000  # ms

    return avg_latency

# endregion


# ==============================================================================
# region BENCHMARK
# ==============================================================================

def benchmark(
    model: nn.Module,
    inputs: Any,
    num_runs: int = 10,
    copy: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Measure and log the complexity of a model.

    Args:
        model (nn.Module): PyTorch model to benchmark.
        inputs (Any): Dummy inputs to the model (e.g., a tensor or a dict of tensors).
        num_runs (int, optional): Number of runs for latency measurement.
            Defaults to 10.
        copy (bool, optional): Whether to make a copy of the model before profiling.
            Defaults to True.
        verbose (bool, optional): Whether to log the results. Defaults to True.

    Returns:
        dict[str, float]: A dictionary containing the measured parameters, MACs,
            FLOPs, and latency.
    """
    # Compute complexity stats
    params, macs, flops = compute_model_stats(model=model, inputs=inputs, copy=copy)

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

def create_dummy_image(imgsz: SizeLike = 512, device = torch.device("cpu")) -> Tensor:
    """Create a dummy image tensor for benchmarking.

    Args:
        imgsz (SizeLike, optional): Image size (e.g., 512 or (512, 512)).
            Defaults to 512.
        device (torch.device, optional): Device to create the tensor on.
            Defaults to torch.device("cpu").

    Returns:
        Tensor: A dummy image tensor of shape (1, 3, H, W) and values ranging
            from 0.0 to 1.0.
    """
    # Normalize imgsz
    imgsz = Size.from_value(imgsz)

    # Read dummy image (e.g., Lenna)
    image = read_image(str(K.DUMMY_IMAGE))
    image = cv2.resize(image, dsize=imgsz.wh)
    image = to_image_tensor(image, normalize=True).to(device)
    return image


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
