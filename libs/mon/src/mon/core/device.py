#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CPU and CUDA device management utilities.

This module provides utilities for listing devices, parsing device specifiers,
normalizing device objects, and querying system and CUDA memory and model device
placement.
"""

from __future__ import annotations

__all__ = [
    "create_device",
    "inspect_model_device",
    "list_devices",
    "parse_device",
    "pynvml_available",
    "query_ram_usages",
    "query_vram_usage",
]

import re

import psutil
import torch
import torch.nn as nn

from mon.core.console import log
from mon.core.enum import MemoryUnit
from mon.core.utils import create_combinations

try:
    import pynvml
    from pynvml.smi import NVMLError
    pynvml_available = True
except ImportError:
    pynvml           = None
    pynvml_available = False
    # Define a placeholder for the exception if pynvml is not installed
    class NVMLError(Exception):
        pass


# ==============================================================================
# region CONSTANTS
# ==============================================================================

CUDA_PREFIX = "cuda:"

# Pre-compile regex for cleaning device strings to improve performance.
# Inside [], ( and ) are literals and do not need escaping.
_DEVICE_CLEAN_RE = re.compile(r"[\[\]()\s'\"]")

# endregion


# ==============================================================================
# region DISCOVERY
# ==============================================================================

def list_devices() -> list[str]:
    """List available device specifiers.

    Returns:
         A list containing "auto", "cpu", "mps" (if available), and all
         available CUDA device specifiers and combinations if CUDA is available.
    """
    devices = ["auto", "cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    
    if torch.cuda.is_available():
        num_devices  = torch.cuda.device_count()
        # Add primary indices first (fastest path)
        cuda_indices = list(range(num_devices))
        devices.extend([f"cuda:{i}" for i in cuda_indices])

        # Only create combinations if more than 1 GPU exists
        if num_devices > 1:
            cuda_combinations = create_combinations(cuda_indices)
            # Filter out combinations of length 1 as we already added them
            devices.extend([
                f"{CUDA_PREFIX}{','.join(map(str, comb))}"
                for comb in cuda_combinations if len(comb) > 1
            ])
    return devices

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_device(device: torch.device | str | int | None) -> torch.device | str:
    """Create a torch.device object from a flexible device input.

    This function acts as a factory, converting various device specifiers into
    a final `torch.device` object or the special "auto" string for libraries
    like PyTorch Lightning.

    Args:
        device: Device specifier, such as a `torch.device` object, an integer,
            a string like "cuda:0", or "auto".

    Returns:
        A `torch.device` object or the special string "auto".

    Raises:
        ValueError: If the ``device`` specifier is unsupported.
    """
    if isinstance(device, torch.device):
        return device

    parsed = parse_device(device)

    if isinstance(parsed, torch.device):
        return parsed

    if parsed == "auto":  # Common in libraries like PyTorch Lightning
        return parsed
    elif parsed == "cuda":
        return torch.device("cuda")
    elif parsed == "cpu":
        return torch.device("cpu")
    elif parsed == "mps":
        return torch.device("mps")
    elif isinstance(parsed, list):  # Handle lists of device indices
        if len(parsed) > 1:
            log(f"Device    : {parsed[0]} is used among {parsed}.")
        else:
            log(f"Device    : {parsed[0]} is used.")
        return torch.device(f"cuda:{parsed[0]}")
    else:
        raise ValueError(f"Unsupported 'device': {device}.")

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---
def inspect_model_device(model: nn.Module) -> torch.device:
    """Inspect the model parameters and return the device used by the first
    parameter.

    This is a reliable way to determine where a model is located.

    Args:
        model: The model whose parameter device is queried.

    Returns:
        The device where the ``model``'s parameters reside. Defaults to "cpu" if
        the ``model`` has no parameters or buffers.
    """
    try:
        # Check parameters first
        return next(model.parameters()).device
    except StopIteration:
        # Fallback to checking buffers (e.g., for BatchNorm running stats)
        try:
            return next(model.buffers()).device
        except StopIteration:
            # If no parameters or buffers, default to CPU
            return torch.device("cpu")


def query_vram_usage(
    device: int        = 0,
    unit  : MemoryUnit = MemoryUnit.GB
) -> tuple[float, float, float]:
    """Query NVML for the specified CUDA device and return memory totals in the
    requested unit.

    Args:
        device: CUDA device index to query.
        unit: Unit to report memory in.

    Returns:
        A tuple of (total, used, free) VRAM values in the requested unit.
        
    Raises:
        ImportError: If ``pynvml`` is not installed.
        NVMLError: If there is an error communicating with the NVIDIA driver.
    """
    if not pynvml_available:
        raise ImportError("Please install 'nvidia-ml-py3' to use 'pynvml'.")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Cache the ratio to avoid repeated lookups
        ratio  = MemoryUnit.names_to_bytes()[MemoryUnit(unit)]
        return info.total / ratio, info.used / ratio, info.free / ratio
    finally:
        # Crucial: Always shut down NVML to release driver handles, even if errors occur.
        try:
            pynvml.nvmlShutdown()
        except NVMLError:
            # This can happen if the driver is already shut down or unavailable.
            pass


def query_ram_usages(unit: MemoryUnit = MemoryUnit.GB) -> tuple[float, float, float]:
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


def parse_device(
    device: torch.device | str | int | None
) -> torch.device | str | list[str]:
    """Parse a device input into a canonical representation.

    This function handles various device formats, including torch.device objects,
    integers, strings (e.g., "cpu", "cuda", "cuda:0,1"), and None.

    Args:
        device: The device specifier to parse.

    Returns:
        A canonical representation: "cpu", "auto", a torch.device, or a
        list of CUDA indices as strings.
    """
    if isinstance(device, torch.device):
        return device
    
    if device is None:
        return "cpu"
    
    if isinstance(device, int):
        return [str(device)]
    
    if isinstance(device, str):
        device = device.lower().strip()
        if not device or device == "cpu":
            return "cpu"
        if device in ("auto", "cuda", "mps"):
            return device
        if device.startswith("cpu") or device.startswith("mps"):
            return torch.device(device)
        
        # Clean the string by removing brackets, spaces, quotes, and "cuda:" prefix
        clean_str = _DEVICE_CLEAN_RE.sub("", device).replace("cuda:", "")
        return [x for x in clean_str.split(",") if x]

    return device


# --- Selection ---


# --- Aggregation ---


# endregion
