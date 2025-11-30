#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for device management.

This module provides utility functions for managing and querying computational
devices, including CPU and GPU (CUDA) devices.
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
    """Lists available devices for computation.
    
    Returns:
        list[str]: A list of device strings, including "auto", "cpu", and
        available CUDA device combinations (e.g., "cuda:0", "cuda:1",
        "cuda:0,1", etc.).
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
    """Retrieves CUDA memory status as a tuple of (total, used, free) memory.
    
    Args:
        device (int): CUDA device index. Defaults to 0.
        unit (MemoryUnit): Memory unit. Defaults to MemoryUnit.GB.
        
    Returns:
        tuple[int, int, int]: A tuple of (total, used, free) memory values in
            the specified unit.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.name_to_byte()[unit]
    return (
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    )


def get_memory_usages(unit: MemoryUnit = MemoryUnit.GB) -> tuple[int, int, int]:
    """Retrieves system memory status as a tuple of (total, used, free) memory.
    
    Args:
        unit (MemoryUnit): Memory unit. Defaults to MemoryUnit.GB.
        
    Returns:
        tuple[int, int, int]: A tuple of (total, used, free) memory values in
            the specified unit.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.name_to_byte()[MemoryUnit(unit)]
    return (
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    )


def get_model_device(model: nn.Module) -> torch.device:
    """Retrieves the allocated device of a PyTorch model.
    
    Args:
        model (nn.Module): A PyTorch model.
        
    Returns:
        torch.device: The device where the model's parameters are allocated.
    """
    return next(model.parameters()).device


# ----- Update -----
def create_device(device: Any) -> torch.device | str:
    """Creates a torch.device instance from the given device input.
    
    Args:
        device (Any): Device input to create a torch.device from.
        
    Returns:
        A torch.device instance, or a device string (e.g., "auto", "cpu", or "cuda").
    
    Raises:
        ValueError: If the device input is unknown.
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
def parse_device(device: Any) -> Union[torch.device, str, list[str]]:
    """Parses the device input into a standardized format.

    Args:
        device (Any): Device input to create a torch.device from.
        
    Returns:
        Union[torch.device, str, list[str]]: Parsed device representation. It
            can be:
            
            - A torch.device instance.
            - A device string (e.g., "auto", "cpu", or "cuda") for torch.device().
            - A list of CUDA device index strings (e.g., ['0', '1']) for distributed training.
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
