#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model complexity metrics.

This module provides functions to compute the number of parameters, MACs, and
FLOPs of a given PyTorch model. It also includes a benchmarking function to log
these statistics.
"""

__all__ = [
    "benchmark",
    "compute_model_stats",
]

import copy

import thop
import torch
import torch.nn as nn

from mon.core import inspect_model_device, image as I, log


# ==============================================================================
# MODEL COMPLEXITY METRICS
# ==============================================================================

# --- Complexity Calculation ---
def compute_model_stats(
    model   : nn.Module,
    imgsz   : int = 512,
    channels: int = 3
) -> tuple[float, tuple, float]:
    """Compute the number of parameters, MACs, and FLOPs of a model.
    
    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Defaults to 512.
        channels: Number of input channels. Defaults to 3.
        
    Returns:
        - params: Number of parameters in the model.
        - macs: Multiply-Accumulate Operations of the model.
        - flops: Floating Point Operations of the model.
    """
    h, w         = I.imgsz(imgsz)
    device       = inspect_model_device(model)
    input        = torch.randn(1, channels, h, w).to(device)
    model_copy   = copy.deepcopy(model)
    model_copy   = model_copy.to(device)
    macs, params = thop.profile(model_copy, inputs=(input,), verbose=False)
    flops        = 2 * macs  # FLOPs = 2 * MACs
    # params       = sum(p.numel() for p in model_copy.parameters())
    del model_copy
    
    return params, macs, flops


# --- Reporting ---
def benchmark(model: nn.Module, imgsz: int = 512, channels: int = 3):
    """Measure and log the complexity of a model.

    Args:
        model: PyTorch model to benchmark.
        imgsz: Input image size. Defaults to 512.
        channels: Number of input channels. Defaults to 3.
    """
    params, macs, flops = compute_model_stats(model=model, imgsz=imgsz, channels=channels)
    log(f"Params    : {params:.4f}")
    log(f"MACs      : {macs:.4f}")
    log(f"FLOPs     : {flops:.4f}")
