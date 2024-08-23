#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Strategy.

This module implements strategies used during training machine learning models.
A strategy is a composition of one Accelerator, one Precision Plugin, a
CheckpointIO plugin, and other optional plugins such as the ClusterEnvironment.

References:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
"""

from __future__ import annotations

__all__ = [
    # Accelerator
    "Accelerator",
    "CPUAccelerator",
    "CUDAAccelerator",
    "MPSAccelerator",
    "XLAAccelerator",
    # Strategy
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "ParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy",
    "XLAStrategy",
]

import os
import platform
from typing import Callable

import torch
import torch.cuda
from lightning.pytorch import accelerators, strategies
from torch import distributed

from mon import core
from mon.globals import ACCELERATORS, STRATEGIES

console = core.console


# region Accelerator

Accelerator     = accelerators.Accelerator
CPUAccelerator  = accelerators.CPUAccelerator
CUDAAccelerator = accelerators.CUDAAccelerator
MPSAccelerator  = accelerators.MPSAccelerator
XLAAccelerator  = accelerators.XLAAccelerator

ACCELERATORS.register(name="cpu" , module=CPUAccelerator)
ACCELERATORS.register(name="cuda", module=CUDAAccelerator)
ACCELERATORS.register(name="gpu" , module=CUDAAccelerator)
ACCELERATORS.register(name="mps" , module=MPSAccelerator)
ACCELERATORS.register(name="xla" , module=XLAAccelerator)

# endregion


# region Strategy

Strategy             = strategies.Strategy
DDPStrategy          = strategies.DDPStrategy
DeepSpeedStrategy    = strategies.DeepSpeedStrategy
FSDPStrategy         = strategies.FSDPStrategy
ParallelStrategy     = strategies.ParallelStrategy
SingleDeviceStrategy = strategies.SingleDeviceStrategy
XLAStrategy          = strategies.XLAStrategy

STRATEGIES.register(name = "ddp"          , module = DDPStrategy)
STRATEGIES.register(name = "deepspeed"    , module = DeepSpeedStrategy)
STRATEGIES.register(name = "fsdp"         , module = FSDPStrategy)
STRATEGIES.register(name = "parallel"     , module = ParallelStrategy)
STRATEGIES.register(name = "single_device", module = SingleDeviceStrategy)
STRATEGIES.register(name = "xla"          , module = XLAStrategy)

# endregion


# region Helper Function

def get_distributed_info() -> list[int]:
    """If distributed is available, return the rank and world size, otherwise
    return ``0`` and ``1``.
    
    Returns:
        The rank and world size of the current process.
    """
    if distributed.is_available():
        initialized = distributed.is_initialized()
    else:
        initialized = False
    if initialized:
        rank       = distributed.get_rank()
        world_size = distributed.get_world_size()
    else:
        rank       = 0
        world_size = 1
    return [rank, world_size]


def set_distributed_backend(strategy: str | Callable, cudnn: bool = True):
    """If you're running on Windows, set the distributed backend to gloo. If
    you're running on Linux, set the distributed backend to ``'nccl'``.
    
    Args:
        strategy: The distributed strategy to use. One of: ``'ddp'``, or
            ``'ddp2'``.
        cudnn: Whether to use cuDNN or not. Default: ``True``.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        console.log(f"cuDNN available: [bright_green]True[/bright_green], "
                    f"used:" + "[bright_green]True" if cudnn else "[red]False")
    else:
        console.log(f"cuDNN available: [red]False")
    
    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        if platform.system() == "Windows":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
            console.log(f"Running on a Windows machine, set torch distributed "
                        f"backend to gloo.")
        elif platform.system() == "Linux":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            console.log(f"Running on a Unix machine, set torch distributed "
                        f"backend to nccl.")
            
# endregion
