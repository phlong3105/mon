#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the :mod:`pynvml` package."""

from __future__ import annotations

from pynvml import *

from mon.globals import MemoryUnit


def get_device_memory(
    device: int = 0,
    unit  : MemoryUnit = MemoryUnit.GB
) -> list[int]:
    """Return the GPU memory status as a :class:`tuple` of
    :math:`(total, used, free)`.
    
    Args:
        device: The index of the GPU device. Default: ``0``.
        unit: The memory unit. Default: ``'GB'``.
    """
    nvmlInit()
    unit  = MemoryUnit.from_value(value=unit)
    h     = nvmlDeviceGetHandleByIndex(index=device)
    info  = nvmlDeviceGetMemoryInfo(h)
    ratio = MemoryUnit.byte_conversion_mapping()[unit]
    total = info.total / ratio
    free  = info.free / ratio
    used  = info.used / ratio
    return [total, used, free]
