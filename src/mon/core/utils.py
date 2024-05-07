#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements various useful functions and data structures."""

from __future__ import annotations

__all__ = [
    "check_installed_package",
    "get_gpu_device_memory",
    "is_rank_zero",
    "list_cuda_devices",
    "list_devices",
    "make_divisible",
    "parse_device",
    "parse_hw",
    "set_device",
    "set_random_seed",
    "upcast",
]

import importlib
import math
import os
import random
from typing import Any

import numpy as np
import pynvml
import torch

from mon.globals import MemoryUnit


# region Parsing Ops

def make_divisible(input: Any, divisor: int = 32) -> int | tuple[int, int]:
    """Make an image divisible by a given stride.
    
    Args:
        input: An image size, size, or shape.
        divisor: The divisor. Default: ``32``.
    
    Returns:
        A new image size.
    """
    h, w = parse_hw(input)
    h    = int(math.ceil(h / divisor) * divisor)
    w    = int(math.ceil(w / divisor) * divisor)
    return h, w


def upcast(input: torch.Tensor | np.ndarray, keep_type: bool = False) -> torch.Tensor | np.ndarray:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.
    
    Args:
        input: An input of type :class:`numpy.ndarray` or :class:`torch.Tensor`.
        keep_type: If True, keep the same type (int32  -> int64). Else upcast to
            a higher type (int32 -> float32).
            
    Return:
        An image of higher type.
    """
    if input.dtype is torch.float16:
        return input.to(torch.float32)
    elif input.dtype is torch.float32:
        return input  # x.to(torch.float64)
    elif input.dtype is torch.int8:
        return input.to(torch.int16) if keep_type else input.to(torch.float16)
    elif input.dtype is torch.int16:
        return input.to(torch.int32) if keep_type else input.to(torch.float32)
    elif input.dtype is torch.int32:
        return input  # x.to(torch.int64) if keep_type else x.to(torch.float64)
    elif type(input) is np.float16:
        return input.astype(np.float32)
    elif type(input) is np.float32:
        return input  # x.astype(np.float64)
    elif type(input) is np.int16:
        return input.astype(np.int32) if keep_type else input.astype(np.float32)
    elif type(input) is np.int32:
        return input  # x.astype(np.int64) if keep_type else x.astype(np.int64)
    return input


def parse_hw(size: int | list[int]) -> list[int]:
    """Casts a size object to the standard :math:`[H, W]`.

    Args:
        size: A size of an image, windows, or kernels, etc.

    Returns:
        A size in :math:`[H, W]` format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            if size[0] >= size[3]:
                size = size[0:2]
            else:
                size = size[1:3]
        elif len(size) == 1:
            size = [size[0], size[0]]
    elif isinstance(size, int | float):
        size = [size, size]
    return size


def parse_device(device: Any) -> list[int] | int | str:
    if isinstance(device, torch.device):
        return device
    
    device = device or None
    if device in [None, "", "cpu"]:
        device = "cpu"
    elif device in ["mps", "mps:0"]:
        device = device
    elif isinstance(device, int):
        device = [device]
    elif isinstance(device, str):  # Not ["", "cpu"]
        device = device.lower()
        for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
            device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
        if "," in device:
            device = [int(x) for x in device.split(",")]
        device = [0] if len(device) == 0 else device
    return device

# endregion


# region Device

def list_cuda_devices() -> str | None:
    """List all available cuda devices in the current machine."""
    if torch.cuda.is_available():
        cuda_str    = "cuda:"
        num_devices = torch.cuda.device_count()
        # gpu_devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        for i in range(num_devices):
            cuda_str += f"{i},"
        if cuda_str[-1] == ",":
            cuda_str = cuda_str[:-1]
        return cuda_str
    return None


def list_devices() -> list[str]:
    """List all available devices in the current machine."""
    devices: list[str] = []
    
    # Get CPU device
    devices.append("auto")
    devices.append("cpu")
    
    # Get GPU devices if available
    if torch.cuda.is_available():
        # All GPU devices
        all_cuda_str = "cuda:"
        num_devices  = torch.cuda.device_count()
        # gpu_devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        for i in range(num_devices):
            all_cuda_str += f"{i},"
            devices.append(f"cuda:{i}")
       
        if all_cuda_str[-1] == ",":
            all_cuda_str = all_cuda_str[:-1]
        if all_cuda_str != "cuda:0":
            devices.append(all_cuda_str)
        
    return devices


def set_device(device: Any, use_single_device: bool = True) -> torch.device:
    """Set a cuda device in the current machine.
    
    Args:
        device: Cuda devices to set.
        use_single_device: If ``True``, set a single-device cuda device in the list.
    
    Returns:
        A cuda device in the current machine.
    """
    device = parse_device(device)
    device = device[0] if isinstance(device, list) and use_single_device else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    return device

# endregion


# region DDP (Distributed Data Parallel)

def is_rank_zero() -> bool:
    """From Pytorch Lightning Official Document on DDP, we know that PL
    intended call the main script multiple times to spin off the child
    processes that take charge of GPUs.

    They used the environment variable "LOCAL_RANK" and "NODE_RANK" to denote
    GPUs. So we can add conditions to bypass the code blocks that we don't want
    to get executed repeatedly.
    """
    return True if (
        "LOCAL_RANK" not in os.environ.keys() and
        "NODE_RANK"  not in os.environ.keys()
    ) else False

# endregion


# region NVML (NVIDIA Management Library)

def get_gpu_device_memory(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Return the GPU memory status as a :class:`tuple` of :math:`(total, used, free)`.
    
    Args:
        device: The index of the GPU device. Default: ``0``.
        unit: The memory unit. Default: ``'GB'``.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(value=unit)
    h     = pynvml.nvmlDeviceGetHandleByIndex(index=device)
    info  = pynvml.nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping()[unit]
    total = info.total / ratio
    free  = info.free / ratio
    used  = info.used / ratio
    return [total, used, free]

# endregion


# region Seed

def set_random_seed(seed: int | list[int] | tuple[int, int]):
    """Set random seeds."""
    if isinstance(seed, list | tuple):
        if len(seed) == 2:
            seed = random.randint(seed[0], seed[1])
        else:
            seed = seed[-1]
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# endregion


# region Installed Package

def check_installed_package(package_name: str, verbose: bool = False) -> bool:
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"`{package_name}` is installed.")
        return True
    except ImportError:
        if verbose:
            print(f"`{package_name}` is not installed.")
        return False

# endregion
