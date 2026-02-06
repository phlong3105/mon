#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Device management utilities.

This module provides utilities for listing, parsing, and querying devices.
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
from mon.core.utils import create_combinations, DeviceType, float_3_t

try:
    import pynvml
    from pynvml.smi import NVMLError

    pynvml_available = True
except ImportError:
    pynvml = None
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
    """List available device specifiers that the current environment supports.

    Returns:
        list[str]: List of supported device specifiers.
        Example: ["cpu", "cuda:0", "cuda:1", "cuda:0,1"].
    """
    devices = ["auto", "cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        # Add primary indices first (fastest path)
        cuda_indices = list(range(num_devices))
        devices.extend([f"cuda:{i}" for i in cuda_indices])

        # Only create combinations if more than 1 GPU exists
        if num_devices > 1:
            cuda_combinations = create_combinations(cuda_indices)
            # Filter out combinations of length 1 as we already added them
            devices.extend(
                [
                    f"{CUDA_PREFIX}{','.join(map(str, comb))}"
                    for comb in cuda_combinations if len(comb) > 1
                ],
            )
    return devices


# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_device(device: DeviceType) -> torch.device | str:
    """Create a torch.device object from a device specifier.

    Args:
        device (DeviceType): Device specifier to convert to a torch.device object.

    Returns:
        torch.device | str: Corresponding torch.device object or "auto" string.

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
    """Return the device of a model.

    Inspect the model parameters and return the device used by the first
    parameter. This is a reliable way to determine where a model is located.

    Args:
        model (nn.Module): Model to inspect.

    Returns:
        torch.device: Device of the model.
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


def query_vram_usage(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> float_3_t:
    """Query NVML for the specified CUDA device memory usage.

    Args:
        device (int): CUDA device index to query. Defaults to 0.
        unit (MemoryUnit): Unit to report memory in. Defaults to MemoryUnit.GB.

    Returns:
        tuple[float, float, float]: (total, used, free) memory values in the
            requested unit.

    Raises:
        ImportError: If ``pynvml`` is not installed.
        NVMLError: If there is an error communicating with the NVIDIA driver.
    """
    if not pynvml_available:
        raise ImportError("Please install 'nvidia-ml-py3' to use 'pynvml'.")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Cache the ratio to avoid repeated lookups
        ratio = MemoryUnit.names_to_bytes()[MemoryUnit(unit)]
        return info.total / ratio, info.used / ratio, info.free / ratio
    finally:
        # Crucial: Always shut down NVML to release driver handles, even if
        # errors occur.
        try:
            pynvml.nvmlShutdown()
        except NVMLError:
            # This can happen if the driver is already shut down or unavailable.
            pass


def query_ram_usages(unit: MemoryUnit = MemoryUnit.GB) -> float_3_t:
    """Query system RAM usage.

    Args:
        unit (MemoryUnit): Unit to report memory in. Defaults to MemoryUnit.GB.

    Returns:
        tuple[float, float, float]: (total, used, free) memory values in the
            requested unit.
    """
    memory = psutil.virtual_memory()
    ratio = MemoryUnit.names_to_bytes()[MemoryUnit(unit)]
    return (
        memory.total / ratio,  # total
        memory.used / ratio,  # used
        memory.available / ratio,  # free
    )


def parse_device(device: DeviceType) -> torch.device | str | list[str]:
    """Parse a device input into a canonical representation.

    Handle various device formats, including torch.device objects, integers,
    strings (e.g., "cpu", "cuda", "cuda:0,1"), and None.

    Args:
        device (DeviceType): Device specifier to parse.

    Returns:
        torch.device | str | list[str]: Parsed device representation.
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

        # Clean the string by removing brackets, spaces, quotes, and "cuda:"
        # prefix
        clean_str = _DEVICE_CLEAN_RE.sub("", device).replace("cuda:", "")
        return [x for x in clean_str.split(",") if x]

    return device


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
