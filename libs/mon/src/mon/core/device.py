#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CPU and CUDA device management utilities.

This module provides utilities for listing devices, parsing device specifiers,
normalizing device objects, and querying system and CUDA memory and model device
placement.
"""

__all__ = [
    "create_device",
    "inspect_model_device",
    "list_devices",
    "parse_device",
    "pynvml_available",
    "query_ram_usages",
    "query_vram_usage",
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


# ==============================================================================
# DISCOVERY & INSPECTION
# ==============================================================================

# --- Hardware Enumeration (Listing what exists on the system) ---
def list_devices() -> list[str]:
    """List available device specifiers.

    Returns:
         A list containing "auto", "cpu", and all available CUDA device
        specifiers and combinations if CUDA is available.
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


# --- Telemetry (Querying RAM/VRAM usage) ---
def query_vram_usage(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> tuple[int, int, int]:
    """Query NVML for the specified CUDA device and return memory totals in the
    requested unit.

    Args:
        device: CUDA device index to query.
        unit: Unit to report memory in.

    Returns:
        A tuple of (total, used, free) VRAM values in the requested unit.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.names_to_bytes()[unit]
    return (
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    )


def query_ram_usages(unit: MemoryUnit = MemoryUnit.GB) -> tuple[int, int, int]:
    """Query system RAM usage and return totals in the requested unit.

    Args:
        unit: Unit to report memory in.

    Returns:
        A tuple of (total, used, free) RAM values in the requested unit.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.names_to_bytes()[MemoryUnit(unit)]
    return (
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    )


# --- State Inspection (Checking where a model currently lives) ---
def inspect_model_device(model: nn.Module) -> torch.device:
    """Inspect the model parameters and return the device used by the first
    parameter.

    Args:
        model: The model whose parameter device is queried.

    Returns:
        The device where the model's parameters reside.
    """
    return next(model.parameters()).device


# ==============================================================================
# RESOLUTION & NORMALIZATION
# ==============================================================================

# --- Parsing (String/Int to Intermediate representation) ---
def parse_device(device: Any) -> Union[torch.device, str, list[str]]:
    """Parse device input into a canonical representation.

    Args:
        device: Accept torch.device, integers, common strings, and
            comma-separated device lists and convert them into a canonical form
            suitable for downstream use.

    Returns:
        The parsed device representation: "cpu", "auto", a torch.device, or a
        list of CUDA indices as strings.
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


# --- Factory (Final conversion to torch.device objects) ---
def create_device(device: Any) -> torch.device | str:
    """Convert device input into a normalized representation suitable for
    downstream use.

    Args:
        device: Device specifier in supported forms, such as a device object,
            integer, or a string like "cuda:0" or "auto".

    Returns:
        Either a torch.device or the special string "auto" when applicable.

    Raises:
        ValueError: If an unsupported ``device`` value is provided.
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


# ==============================================================================
# RESOURCE CLEANUP
# ==============================================================================

# --- Cache Management (Emptying CUDA cache, closing NVML) ---
