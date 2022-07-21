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
from one.core.types import MemoryUnit_


# MARK: - Functional

def extract_device_dtype(tensors: list) -> tuple[torch.device, torch.dtype]:
    """
    Take a list of tensors and returns a tuple of `device` and `dtype` that
    are the same for all tensors in the list.
    
    Args:
        tensors (list): list.
    
    Returns:
        A tuple of the device and dtype of the tensor.
    """
    device, dtype = None, None
    for tensors in tensors:
        if tensors is not None:
            if not isinstance(tensors, (Tensor,)):
                continue
            _device = tensors.device
            _dtype  = tensors.dtype
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
    device_index: int = 0, unit: MemoryUnit_ = MemoryUnit.GB
) -> tuple[int, int, int]:
    """
    Return the total, used, and free memory of the GPU with the given index
    in the given unit.
    
    Args:
        device_index (int): The index of the GPU you want to get the memory
            usage of. Defaults to 0.
        unit (MemoryUnit_): MemoryUnit_ = MemoryUnit.GB.
    
    Returns:
        A tuple of the total, used, and free memory in the specified unit.
    """
    nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    h     = nvmlDeviceGetHandleByIndex(device_index)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return total, used, free


def select_device(
    model_name: str        = "",
    device    : str | None = "",
    batch_size: int | None = None
) -> torch.device:
    """
    Return a torch.device object, which is either cuda:0 or cpu, depending
    on whether CUDA is available.
    
    Args:
        model_name (str): The name of the model. This is used to print a
            message to the console.
        device (str | None): The device to run the model on. If None, the
          default device is used.
        batch_size (int | None): The number of samples to process in a single
            batch.
    
    Returns:
        A torch.device object.
    """
    from .rich import console
    device      = f"{device}"
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
    model_name: str        = "",
    device    : str | None = "",
    batch_size: int | None = None
) -> torch.device:
    """
    Set the environment variable `CUDA_VISIBLE_DEVICES` to the requested
    device, and returns a `torch.device` object.
    
    Args:
        model_name (str): The name of the model. This is used to print a message
            to the console.
        device (str | None): The device to run the model on. If None, the
          default device is used.
        batch_size (int | None): The number of samples to process in a single
            batch.
    
    Returns:
        A torch.device object.
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
    """
    Synchronize the CUDA device if it's available, and then returns the
    current time.
    
    Returns:
        The time in seconds since the epoch as a floating point number.
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
