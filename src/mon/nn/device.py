#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements device management functions using PyTorch."""

from __future__ import annotations

__all__ = [
    "extract_device_dtype", "select_device", "time_synchronized",
]

import os
import time
from typing import Any

import torch

from mon.core import console


# region Device Selection

def extract_device_dtype(
    tensors: list[torch.Tensor]
) -> tuple[torch.device, torch.dtype]:
    """Extract the device and data-type from a :class:`list` of tensors.
    
    Args:
        tensors: A :class:`list` of tensors.
    
    Returns:
        A :class:`tuple` of (device, dtype), where device is a
        :class:`torch.device`, and dtype is a :class:`torch.dtype`.
    """
    device, dtype = None, None
    for tensors in tensors:
        if tensors is not None:
            if not isinstance(tensors, torch.Tensor):
                continue
            _device = tensors.device
            _dtype  = tensors.dtype
            if device is None and dtype is None:
                device = _device
                dtype  = _dtype
            
            if device != _device or dtype != _dtype:
                raise ValueError(
                    f"Passed values must be in the same 'device' and 'dtype'. "
                    f"But got: ({device}, {dtype}) and ({_device}, {_dtype})."
                )
                
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype  = torch.get_default_dtype()
    return device, dtype


def select_device(device: Any = "", batch_size: int = 1) -> torch.device:
    """Return a :class:`torch.device` object depending on whether CUDA is
    available on the current system.
    
    Args:
        device: The device to run the model on. If ``None``, the default
            ``'cpu'`` device is used.
        batch_size: The expected batch size in a single forward pass.
    
    Returns:
        A :class:`torch.device` object.
    """
    if device is None:
        return torch.device("cpu")
    elif isinstance(device, int | float):
        device = f"{device}"
    assert isinstance(device, str)

    # CPU
    cpu_request = device.lower() in ["cpu", "default"]
    if cpu_request:
        console.log(f"Using CPU\n")
        return torch.device("cpu")
    
    # GPU / CUDA
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable, invalid device {device} requested.")
    cuda_request = False if cpu_request else torch.cuda.is_available()
    if cuda_request:
        bytes_to_mb = 1024 ** 2  # bytes to MB
        num_gpus    = torch.cuda.device_count()
        # Check if batch_size is compatible with device_count
        if num_gpus > 1 and batch_size > 1:
            if batch_size % num_gpus != 0:
                raise ValueError(
                    f":param:'batch_size' must be a multiple of GPU counts "
                    f"{num_gpus}. But got: {batch_size} % {num_gpus} != 0."
                )
                
        x = [torch.cuda.get_device_properties(i) for i in range(num_gpus)]
        s = "Using CUDA "
        for i in range(0, num_gpus):
            if i == 1:
                s = " " * len(s)
            console.log(
                "%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / bytes_to_mb)
            )
        # Here we select the first cuda device by default
        os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
        console.log(f"")
        return torch.device("cuda:0")
    
    # If nothing works, just use CPU
    console.log(f"Using CPU\n")
    return torch.device("cpu")

# endregion


# region Device Synchronization

def time_synchronized(device: Any = None) -> float:
    """Wait for all kernels in all streams on a CUDA device to complete, and
    then return the current time.
    
    Args:
        device: a device for which to synchronize. If ``None``, it uses the
        current device, given by :meth:`current_device()`. Default: ``None``.
    """
    torch.cuda.synchronize(device=device) if torch.cuda.is_available() else None
    return time.time()

# endregion
