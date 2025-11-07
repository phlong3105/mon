#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for managing and querying computational
devices, including CPU and GPU (CUDA) devices.

It includes functions to list available devices, retrieve memory usage statistics,
determine the device of a model, and create device instances for PyTorch operations.
"""

__all__ = [
    "create_device",
    "get_cuda_memory_usages",
    "get_memory_usages",
    "get_model_device",
    "list_devices",
    "parse_device",
    "pynvml_available",
]

from typing import Any, Union

import psutil
import torch
import torch.nn as nn

from mon.core.console import log
from mon.core.enum import MemoryUnit
from mon.core.utils import create_combinations

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False

CUDA_PREFIX = "cuda:"


# ----- Retrieve -----
def list_devices() -> list[str]:
    """Lists all available devices on the current machine.

    Returns:
        A ``list`` of device strings including ``auto``, ``cpu``, and CUDA
        devices if available.
    """
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices <= 0:
            return devices
        # Add all CUDA device combinations (e.g., ``cuda:0``, ``cuda:1``, ``cuda:0,1``, etc.)
        cuda_indices      = list(range(num_devices))
        cuda_combinations = create_combinations(cuda_indices)
        devices.extend([f"{CUDA_PREFIX}{','.join(str(i) for i in comb)}" for comb in cuda_combinations])
    return devices


def get_cuda_memory_usages(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> tuple[int, int, int]:
    """Retrieves GPU memory status as a ``tuple`` of :math:`(total, used, free)`
    memory.

    Args:
        device: GPU device index. Default: ``0``.
        unit: Memory unit (e.g., ``GB``). Default: ``MemoryUnit.GB``.

    Returns:
        A ``tuple`` of :math:`(total, used, free)` memory values in the
        specified ``unit``.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.name_to_byte()[unit]
    return (
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    )


def get_memory_usages(unit: MemoryUnit = MemoryUnit.GB) -> tuple[int, int, int]:
    """Retrieves RAM status as a list of :math:`(total, used, free)` memory.

    Args:
        unit: Memory unit (e.g., ``GB``). Default: ``MemoryUnit.GB``.

    Returns:
        A ``tuple`` of :math:`(total, used, free)` memory values in the
        specified ``unit``.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.name_to_byte()[MemoryUnit.from_value(unit)]
    return (
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    )


def get_model_device(model: nn.Module) -> torch.device:
    """Gets the current device of a model.

    Args:
        model: The model to check.

    Returns:
        A ``torch.device`` instance where model parameters reside.
    """
    return next(model.parameters()).device


# ----- Update -----
def create_device(device: Any) -> Union[torch.device, str]:
    """Create a device for the current process.
    
    Args:
        device: Device to set (e.g., CUDA device index(es) or string).
    
    Returns:
        A ``torch.device`` instance, defaults to ``torch.device("cpu")``.
    """
    if isinstance(device, torch.device):
        return device

    device = parse_device(device)

    if device == "auto":  # Used in PyTorch Lighting's Trainer.
        return device
    elif device == "cuda":
        return torch.device("cuda")
    elif device == "cpu":
        return torch.device("cpu")
    elif isinstance(device, list):  # Use the first CUDA device.
        log(f"Device    : {device[0]} is used among {device}.")
        return torch.device(f"cuda:{device[0]}")
    else:
        raise ValueError(f"Unknown device: {device}.")


# ----- Convert -----
def parse_device(device: Any) -> torch.device | str | list[str]:
    """Parses device(s) into appropriate formats.

    Args:
        device: Device to parse.
         
    Returns:
        - A ``torch.device`` instance.
        - A device ``str``: ``auto``, ``cpu``, or ``cuda`` for ``torch.device()``.
        - A ``list`` of CUDA device index strings (e.g., ``['0', '1']``) for distributed training.
    """
    if isinstance(device, torch.device):
        return device
    if device in [None, "", "cpu"]:
        return "cpu"
    if device in ["auto", "cuda"]:
        return device

    if isinstance(device, int):
        device = [str(device)]
    if isinstance(device, str):
        device = (device.lower()
                        .replace("cuda:", "")
                        .translate(str.maketrans("", "", "()[ ]' ")))
        device = device.split(",")
        device = [str(i) for i in device]

    return device
