#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Device Management.

This module provides utilities for listing, parsing, and querying devices.
"""

from __future__ import annotations

__all__ = [

]

import re

import psutil
import torch
import torch.nn as nn

from mon.core.enum import MemoryUnit
from mon.core.typing import DeviceLike
from mon.core.utils import create_combinations

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

_MISSING = object()
_CUDA_PREFIX = "cuda:"

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
        # Add primary indices first (the fastest path)
        cuda_indices = list(range(num_devices))
        devices.extend([f"cuda:{i}" for i in cuda_indices])

        # Only create combinations if more than 1 GPU exists
        if num_devices > 1:
            cuda_combinations = create_combinations(cuda_indices)
            # Filter out combinations of length 1 as we already added them
            devices.extend(
                [
                    f"{_CUDA_PREFIX}{','.join(map(str, comb))}"
                    for comb in cuda_combinations if len(comb) > 1
                ],
            )
    return devices

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def create_device(device: DeviceLike = _MISSING) -> torch.device | str:
    """Create a torch.device object from a device specifier.

    Args:
        device (DeviceType): Device specifier.

    Returns:
        torch.device | str: Corresponding torch.device object or "auto" string.

    Raises:
        ValueError: If the ``device`` specifier is unsupported.
    """
    # Fast path for common cases to avoid unnecessary parsing overhead.
    if device is _MISSING:
        return torch.device("cpu")
    elif device is None:
        return torch.device("cpu")
    elif isinstance(device, torch.device):
        return device

    # Parse the device specifier
    parsed = parse_device(device)
    if isinstance(parsed, torch.device):
        return parsed
    elif parsed == "auto":  # Common in libraries like PyTorch Lightning
        return parsed
    elif parsed == "cuda":
        return torch.device("cuda")
    elif parsed == "cpu":
        return torch.device("cpu")
    elif parsed == "mps":
        return torch.device("mps")
    elif isinstance(parsed, list):  # Handle lists of device indices
        return torch.device(f"cuda:{parsed[0]}")
    else:
        raise ValueError(f"Unsupported 'device': {device}.")

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


def parse_device(device: DeviceLike = _MISSING) -> torch.device | str | list[str]:
    """Parse a device input into a canonical representation.

    Handle various device formats, including torch.device objects, integers,
    strings (e.g., "cpu", "cuda", "cuda:0,1"), and None.

    Args:
        device (DeviceType, optional): Device specifier to parse.
            Defaults to _MISSING.

    Returns:
        torch.device | str | list[str]: Parsed device representation.
    """
    # Fast path for common cases to avoid unnecessary parsing overhead.
    if device is _MISSING:
        return "cpu"
    elif device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device

    # Handle integers as CUDA device indices (e.g., 0 -> "cuda:0")
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
        clean_str = _DEVICE_CLEAN_RE.sub("", device).replace(_CUDA_PREFIX, "")
        return [x for x in clean_str.split(",") if x]

    return [str(d).lower().strip() for d in device]


def parse_device_str(device: DeviceLike = _MISSING) -> torch.device | str:
    """Parse a device input into a canonical representation.

    Handle various device formats, including torch.device objects, integers,
    strings (e.g., "cpu", "cuda", "cuda:0,1"), and None.

    Args:
        device (DeviceType): Device specifier to parse. Defaults to _MISSING.

    Returns:
        torch.device | str: Parsed device representation.
    """
    # Fast path for common cases to avoid unnecessary parsing overhead.
    if device is _MISSING:
        return "cpu"
    elif device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device

    # Handle integers as CUDA device indices (e.g., 0 -> "cuda:0")
    if isinstance(device, int):
        return f"{_CUDA_PREFIX}{device}"

    # Handle strings as comma-separated lists of device indices
    # (e.g., "0,1" -> ["cuda:0", "cuda:1"])
    if isinstance(device, str):
        device = device.lower().strip()
        if not device:
            return "cpu"
        if device in ("auto", "cpu", "cuda", "mps"):
            return device

        # Clean the string by removing brackets, spaces, quotes, and "cuda:" prefix
        clean_str = _DEVICE_CLEAN_RE.sub("", device).replace(_CUDA_PREFIX, "")
        return [x for x in clean_str.split(",") if x]

    return [str(d).lower().strip() for d in device]


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
