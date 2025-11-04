#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements efficiency score metrics."""

__all__ = [
    "benchmark",
    "compute_model_stats",
]

import copy

import thop
import torch
import torch.nn as nn

from mon.core.console import log
from mon.core.device import get_model_device
from mon.core.dtypes import image as I


def compute_model_stats(
    model   : nn.Module,
    imgsz   : int = 512,
    channels: int = 3
) -> tuple[float, tuple, float]:
    """Computes FLOPs and parameters for a model. Note: 1 MAC ≈ 2 FLOPs
    
    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.
        channels: Number of input channels. Default: ``3``.
    
    Returns:
        A tuple of :math:`(macs, flops, params)`.
    """
    h, w         = I.imgsz(imgsz)
    device       = get_model_device(model)
    input        = torch.randn(1, channels, h, w).to(device)
    model_copy   = copy.deepcopy(model)
    model_copy   = model_copy.to(device)
    macs, params = thop.profile(model_copy, inputs=(input,), verbose=False)
    flops        = 2 * macs  # FLOPs = 2 * MACs
    # params       = sum(p.numel() for p in model_copy.parameters())
    del model_copy
    
    return params, macs, flops


def benchmark(model: nn.Module, imgsz: int = 512, channels: int = 3):
    """Measures and logs the complexity of a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.
        channels: Number of input channels. Default: ``3``.
    """
    params, macs, flops = compute_model_stats(model=model, imgsz=imgsz, channels=channels)
    log(f"Params    : {params:.4f}")
    log(f"MACs      : {macs:.4f}")
    log(f"FLOPs     : {flops:.4f}")
