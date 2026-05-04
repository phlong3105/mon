#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""System Context.

This module provides utilities for managing system-wide dynamic runtime states.

Notes:
    - ``context.py`` is for Dynamic Runtime State. This is the "living"
      environment of your current run. These values are determined at runtime.
    - ``constants.py`` is for static, immutable truths. These are hardcoded
      values that will never change during a single run of your program.
"""

from __future__ import annotations

__all__ = [
    "SystemContext",
    "sys_ctx",
]

import os
import random
from typing import Sequence

import numpy as np
import torch

from .base import singleton
from .data import Device, DeviceList
from .dtype import DeviceType
from .typing import IntOrTuple2, MISSING


# ==============================================================================
# region CONTROL
# ==============================================================================

@singleton
class SystemContext:
    """A singleton class for managing system-wide dynamic runtime states."""

    # --- Lifecycle & Initialization ---
    def __init__(self):
        """Initialize a new instance."""
        # Allocate resources
        self._devices = DeviceList()
        self.scan_devices()

    # --- Properties ---
    @property
    def devices(self) -> DeviceList:
        """Return the list of available devices."""
        return self._devices

    @property
    def device_names(self) -> list[str]:
        """Return a list of all device names."""
        return self.devices.names

    @property
    def cpu(self) -> Device:
        """Return the CPU device."""
        return self.devices["cpu"]

    @property
    def mps(self) -> Device:
        """Return the MPS device if available, otherwise "cpu"."""
        return self.get_device(device="mps")

    @property
    def cudas(self) -> list[Device]:
        """Return a list of all CUDA devices in the system."""
        return [d for d in self.devices.values if d.is_cuda]

    # --- Discovery ---
    def scan_devices(self) -> DeviceList:
        """Scan the system for available devices and populate the ``self._devices``
        property.

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

        self._devices = devices
        return self._devices

    # --- Retrieval ---
    def get_device(self, device: torch.device | str | int = MISSING) -> Device:
        """Return the device object for the specified device.

        Args:
            device (torch.device | str | int, optional): Device to retrieve.
                Defaults to _MISSING.

        Returns:
            Device: The corresponding Device object.

        Raises:
            ValueError: If the ``device`` specifier is unsupported.
        """
        if device is MISSING:
            return self.devices["cpu"]
        elif device is None:
            return self.devices["cpu"]
        elif isinstance(device, torch.device):
            return self.devices.get(str(device), self.devices["cpu"])
        elif isinstance(device, str):
            device_lower = device.lower()

            # Add routing for automatic device selection
            if device_lower in ["auto", "free"]:
                free_device = self.get_free_cuda_device()
                return free_device if free_device is not None else self.devices["cpu"]

            return self.devices.get(device_lower, self.devices["cpu"])
        elif isinstance(device, int):
            key = f"cuda:{device}" if device >= 0 else "cpu"
            return self.devices[key] if key in self.devices else self.devices["cpu"]
        else:
            raise ValueError(f"Unsupported device specifier: '{device}'")

    def get_torch_device(self, device: torch.device | str | int = MISSING) -> torch.device:
        """Return the torch device object for the specified device."""
        return self.get_device(device).torch_device

    def get_free_cuda_device(
        self,
        max_mem_util: float = 0.5,
        max_compute_util: int = 50
    ) -> Device | None:
        """Find the first available CUDA device below the specified utilization
        thresholds.

        Args:
            max_mem_util (float, optional): Maximum allowed VRAM utilization
                ratio (0.0 to 1.0). Defaults to 0.5 (50%).
            max_compute_util (int, optional): Maximum allowed compute utilization
                percentage (0 to 100). Defaults to 50.

        Returns:
            Device | None: The free CUDA device, or None if all are busy/unavailable.
        """
        if not torch.cuda.is_available():
            return None

        for device in self.cudas:
            t_device = torch.device(device.name)

            # 1. Check true OS-level VRAM usage
            # mem_get_info returns (free_memory, total_memory) in bytes
            free_mem, total_mem = torch.cuda.mem_get_info(t_device)
            used_mem = total_mem - free_mem
            mem_util = used_mem / total_mem

            # 2. Check Volatile GPU Compute Utilization
            # returns the percent of time over the past sample period during
            # which one or more kernels was executing on the GPU.
            try:
                compute_util = torch.cuda.utilization(t_device)
            except AttributeError:
                # Fallback in case of older PyTorch versions (< 1.12)
                compute_util = 0

            # If both VRAM and Compute are below the threshold, claim it!
            if mem_util <= max_mem_util and compute_util <= max_compute_util:
                return device

        # All GPUs are heavily utilized
        return None

    # --- Mutation ---
    @staticmethod
    def set_random_seed(seed: IntOrTuple2, deterministic: bool = False):
        """Set random seeds for Python, NumPy, and PyTorch.

        Args:
            seed (IntOr2Tuple): Single seed value or a range of [min, max] from
                which a seed will be randomly sampled.
            deterministic (bool, optional): If True, configures PyTorch for
                deterministic behavior. Defaults to False.
        """
        if isinstance(seed, Sequence):
            # If a range is provided, sample a seed from it.
            seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]

        # Set seeds for all relevant libraries.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if deterministic:
            # Configure PyTorch for deterministic behavior.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Use deterministic algorithms, warning if they are not available.
            torch.use_deterministic_algorithms(True, warn_only=True)


sys_ctx = SystemContext()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
