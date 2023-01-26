#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the :mod:`pynvml` package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pynvml import *

from mon.core import constant

if TYPE_CHECKING:
    from mon.core.typing import MemoryUnitType


def get_gpu_memory(
    device_index: int            = 0,
    unit        : MemoryUnitType = constant.MemoryUnit.GB,
) -> tuple[int, int, int]:
    """Return the GPU memory status as a tuple (total, used, and free).
    
    Args:
        device_index: The index of the GPU device. Defaults to 0.
        unit: The memory unit. Defaults to “GB”.
    """
    nvmlInit()
    unit  = constant.MemoryUnit.from_value(unit)
    h     = nvmlDeviceGetHandleByIndex(device_index)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = constant.MemoryUnit.byte_conversion_mapping()[unit]
    total = info.total / ratio
    free  = info.free  / ratio
    used  = info.used  / ratio
    return total, used, free
