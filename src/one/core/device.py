#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
from time import time

import torch
from pynvml import *
from torch import Tensor

from one.core.types import assert_number_divisible_to
from one.core.types import MemoryUnit


# MARK: - Functional

def extract_device_dtype(tensor_list: list) -> tuple[torch.device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are
    Tensor). If so, it would return a tuple of (device, dtype).
    Default: (cpu, `get_default_dtype()`).

    Returns:
        device (torch.device):
            Device of the tensor.
        dtype (torch.dtype):
        
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (Tensor,)):
                continue
            _device = tensor.device
            _dtype  = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype  = _dtype
            
            if device != _device or dtype != _dtype:
                raise ValueError(
                    f"Passed values must be in the same `device` and `dtype`. "
                    f"But got: ({device}, {dtype}) and ({_device}, {_dtype})."
                )
                
    if device is None:
        # TODO: update this when having torch.get_default_device()
        device = torch.device("cpu")
    if dtype is None:
        dtype  = torch.get_default_dtype()
    return device, dtype


def get_gpu_memory(
    device_index: int = 0,
    unit        : Union["MemoryUnit", str, int] = MemoryUnit.GB
) -> tuple[int, int, int]:
    if isinstance(unit, (str, int)):
        unit = MemoryUnit.from_value(unit)
    
    if unit not in MemoryUnit:
        from one.core import error_console
        error_console.log(f"Unknown memory unit: {unit}.")
        unit = MemoryUnit.GB
        
    nvmlInit()
    h     = nvmlDeviceGetHandleByIndex(device_index)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return total, used, free


def select_device(
    model_name: str              = "",
    device    : Union[str, None] = "",
    batch_size: Union[int, None] = None
) -> torch.device:
    """Select the device to runners the model.
    
    Args:
        model_name (str):
            Name of the model. Default: "".
        device (str, None):
            Name of device for running. Can be: 'cpu' or '0' or '0,1,2,3'.
            Default: "".
        batch_size (int, None):
            Number of samples in one forward & backward pass. Default: `None`.

    Returns:
        device (torch.device):
            GPUs or CPU.
    """
    from .rich import console
    
    if isinstance(device, int):
        device = f"{device}"
        
    cpu_request = device.lower() == "cpu"
    if device and not cpu_request:  # If device requested other than `cpu`
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        if not torch.cuda.is_available():  # Check availability
            raise RuntimeError(
                f"CUDA unavailable, invalid device {device} requested."
            )
            
    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        bytes_to_mb = 1024 ** 2  # bytes to MB
        num_gpus    = torch.cuda.device_count()
        if num_gpus > 1 and batch_size:  # check that batch_size is compatible with device_count
            if batch_size % num_gpus != 0:
                raise ValueError(
                    f"`batch-size` must be a multiple of GPU count {num_gpus}. "
                    f"But got: {batch_size} % {num_gpus} != 0."
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
    else:
        console.log("Using CPU")

    console.log("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def select_device_old(
    model_name: str              = "",
    device    : Union[str, None] = "",
    batch_size: Union[int, None] = None
) -> torch.device:
    """Select the device to runners the model.
    
    Args:
        model_name (str):
            Name of the model. Default: "".
        device (str, None):
            Name of device for running. Default: "".
        batch_size (int, None):
            Number of samples in one forward & backward pass. Default: `None`.

    Returns:
        device (torch.device):
            GPUs or CPU.
    """
    from .rich import console
    
    if device is None:
        return torch.device("cpu")

    # device = 'cpu' or '0' or '0,1,2,3'
    s   = f"{model_name}"  # string
    
    if isinstance(device, str) and device.lower() == "cpu":
        cpu = True
    else:
        cpu = False
    
    if cpu:
        # Force torch.cuda.is_available() = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        # Non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        # Check availability
        if not torch.cuda.is_available():  # Check availability
            raise RuntimeError(
                f"CUDA unavailable, invalid device {device} requested."
            )
    
    cuda = not cpu and torch.cuda.is_available()
    
    if cuda:
        n = torch.cuda.device_count()

        # Check that batch_size is compatible with device_count
        if n > 1 and batch_size:
            assert_number_divisible_to(batch_size, n)
        space = " " * len(s)
        
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"
    
    console.log(s)
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
