#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Device Data Structure.

This module provides data structures for handling devices.
"""

from __future__ import annotations

__all__ = [
    "Device",
    "DeviceList",
    "DeviceManager",
    "inspect_model_device",
    "query_ram_usages",
    "query_vram_usage",
]

from dataclasses import dataclass
from typing import Iterable

import psutil
import torch
from torch import nn

from mon.core.data.structs import IndexList
from mon.core.enum import DeviceType, MemoryUnit
from mon.core.singleton import singleton
from mon.core.typing import DeviceLike

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

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Device:
    """Data structure representing a device.

    Attributes:
        name (str): The name of the device (e.g., "cpu", "cuda:0").
        type (DeviceType): The type of the device (CPU, CUDA, MPS).
        index (int, optional): The index of the device if applicable (e.g., 0
            for "cuda:0"). Defaults to -1.
    """

    name: str
    type: DeviceType
    index: int = -1

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            ValueError: If the device ``type`` is CUDA but the ``index`` is invalid.
        """
        # Validate inputs
        if self.is_cuda and self.index < 0:
            raise ValueError(f"Invalid CUDA device index: {self.index}")

    # --- Representation ---
    def __str__(self) -> str:
        """Informal string representation for end-users (print)."""
        return self.string

    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"type={self.type}, "
            f"index={self.index}"
            f")"
        )

    # --- Properties ---
    @property
    def is_cpu(self) -> bool:
        """Return True if the device is a CPU device."""
        return self.type == DeviceType.CPU

    @property
    def is_cuda(self) -> bool:
        """Return True if the device is a CUDA device."""
        return self.type == DeviceType.CUDA

    @property
    def is_mps(self) -> bool:
        """Return True if the device is an Apple MPS device."""
        return self.type == DeviceType.MPS

    @property
    def torch_device(self) -> torch.device:
        """Return the torch device object."""
        return torch.device(self.__str__())

    @property
    def string(self) -> str:
        """Return the device string representation."""
        if self.is_cuda and self.index >= 0:
            return f"{self.type}:{self.index}"
        else:
            return self.type

    # --- Retrieval ---
    def usages(self, unit: MemoryUnit = "GB") -> tuple[float, float, float]:
        """Return the memory usage (total, used, free) for the device.

        Args:
            unit (MemoryUnit, optional): Unit to report memory in.
                Defaults to MemoryUnit.GB.

        Returns:
            tuple[float, float, float]: (total, used, free) memory values in the
                requested unit.
        """
        if self.is_cuda:
            return query_vram_usage(self.index, unit=unit)
        else:
            return query_ram_usages(unit=unit)


class DeviceList(IndexList[Device]):
    """A list of ``Device`` instances, accessible by index or name.

    Extend ``IndexList`` to provide dictionary-like access to ``Device``
    instances by their ``name`` attribute.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: Iterable | None = None):
        """Initialize a new instance.

        Args:
            data (Iterable, optional): Initial data to populate the list.
                Defaults to None.
        """
        # We hardcode the item_type and key here
        # Users just call ModalityList() without arguments
        super().__init__(item_type=Device, data=data, key="name")

    # --- Properties ---
    @property
    def names(self) -> list[str]:
        """Return a list of names."""
        return self.keys()


@singleton
class DeviceManager:
    """A singleton class for managing all devices in the system."""

    # --- Lifecycle & Initialization ---
    def __init__(self):
        """Initialize a new instance."""
        self.devices = DeviceList()
        self.list_devices()

    # --- Properties ---
    @property
    def all_devices(self) -> list[str]:
        """Return a list of all devices in the system."""
        return self.devices.keys

    # --- Discovery ---
    def list_devices(self) -> DeviceList:
        """List all devices in the current system.

        After calling this method, the ``devices`` property will be populated.

        Returns:
            DeviceList: A list of available devices.
        """
        devices = DeviceList()

        # CPU is always available
        devices.append(Device(name="cpu", type=DeviceType.CPU))

        # Check for MPS (Apple Silicon) support
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(Device(name="mps", type=DeviceType.MPS))

        # Check for CUDA support and list all available CUDA devices
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            for i in range(num_devices):
                devices.append(Device(name=f"cuda:{i}", type=DeviceType.CUDA, index=i))

        self.devices = devices
        return self.devices

    # --- Retrieval ---
    def get_device(self, device: DeviceLike = _MISSING) -> Device:
        """Return the device object for the specified device.

        Args:
            device (DeviceLike, optional): Device to retrieve. Defaults to _MISSING.

        Returns:
            Device: The corresponding Device object.

        Raises:
            ValueError: If the ``device`` specifier is unsupported.
        """
        if device is _MISSING:
            return self.devices["cpu"]
        elif device is None:
            return self.devices["cpu"]
        elif isinstance(device, torch.device):
            return self.devices[str(device)]
        elif isinstance(device, str):
            return self.devices[device]
        elif isinstance(device, int):
            key = f"cuda:{device}" if device >= 0 else "cpu"
            return self.devices[key] if key in self.devices else self.devices["cpu"]
        else:
            raise ValueError(f"Unsupported device specifier: {device}")

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---

def inspect_model_device(model: nn.Module) -> torch.device:
    """Inspect the model parameters and return the device used by the first
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


def query_vram_usage(
    device: int = 0,
    unit: MemoryUnit = "GB"
) -> tuple[float, float, float]:
    """Query NVML for the specified CUDA device memory usage.

    Args:
        device (int, optional): CUDA device index to query. Defaults to 0.
        unit (MemoryUnit, optional): Unit to report memory in. Defaults to "GB".

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
        ratio = MemoryUnit.names_to_bytes()[unit]
        return info.total / ratio, info.used / ratio, info.free / ratio
    finally:
        # Crucial: Always shut down NVML to release driver handles, even if
        # errors occur.
        try:
            pynvml.nvmlShutdown()
        except NVMLError:
            # This can happen if the driver is already shut down or unavailable.
            pass


def query_ram_usages(unit: MemoryUnit = "GB") -> tuple[float, float, float]:
    """Query system RAM usage.

    Args:
        unit (MemoryUnit, optional): Unit to report memory in.
            Defaults to MemoryUnit.GB.

    Returns:
        tuple[float, float, float]: (total, used, free) memory values in the
            requested unit.
    """
    memory = psutil.virtual_memory()
    ratio = MemoryUnit.names_to_bytes()[unit]
    return (
        memory.total / ratio,      # total
        memory.used / ratio,       # used
        memory.available / ratio,  # free
    )


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
