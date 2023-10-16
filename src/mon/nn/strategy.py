#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements strategies used during training machine learning
models. A strategy is a composition of one Accelerator, one Precision Plugin, a
CheckpointIO plugin, and other optional plugins such as the ClusterEnvironment.

References:
    `<https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html>`__
"""

from __future__ import annotations

__all__ = [
    "FSDPStrategy", "CPUAccelerator", "CUDAAccelerator", "ParallelStrategy",
    "SingleDeviceStrategy", "DDPStrategy", "DeepSpeedStrategy",
    "HPUAccelerator", "HPUParallelStrategy", "Strategy", "IPUAccelerator",
    "IPUStrategy", "MPSAccelerator", "SingleHPUStrategy", "SingleTPUStrategy",
    "TPUAccelerator", "XLAStrategy",
]

import os
import platform
from typing import Callable

import torch
import torch.cuda
from lightning.pytorch import accelerators, strategies
from torch import distributed

from mon.core import console
from mon.globals import ACCELERATORS, STRATEGIES


# region Accelerator

CPUAccelerator  = accelerators.CPUAccelerator
CUDAAccelerator = accelerators.CUDAAccelerator
HPUAccelerator  = accelerators.HPUAccelerator
IPUAccelerator  = accelerators.IPUAccelerator
MPSAccelerator  = accelerators.MPSAccelerator
TPUAccelerator  = accelerators.TPUAccelerator

ACCELERATORS.register(name="cpu" , module=CPUAccelerator)
ACCELERATORS.register(name="cuda", module=CUDAAccelerator)
ACCELERATORS.register(name="gpu" , module=CUDAAccelerator)
ACCELERATORS.register(name="hpu" , module=HPUAccelerator)
ACCELERATORS.register(name="ipu" , module=IPUAccelerator)
ACCELERATORS.register(name="mps" , module=MPSAccelerator)
ACCELERATORS.register(name="tpu" , module=TPUAccelerator)

# endregion


# region Strategy

DDPStrategy          = strategies.DDPStrategy
DeepSpeedStrategy    = strategies.DeepSpeedStrategy
FSDPStrategy         = strategies.FSDPStrategy
HPUParallelStrategy  = strategies.HPUParallelStrategy
IPUStrategy          = strategies.IPUStrategy
ParallelStrategy     = strategies.ParallelStrategy
SingleDeviceStrategy = strategies.SingleDeviceStrategy
SingleHPUStrategy    = strategies.SingleHPUStrategy
SingleTPUStrategy    = strategies.SingleTPUStrategy
Strategy             = strategies.Strategy
XLAStrategy          = strategies.XLAStrategy

STRATEGIES.register(name = "ddp"          , module = DDPStrategy)
STRATEGIES.register(name = "deepspeed"    , module = DeepSpeedStrategy)
STRATEGIES.register(name = "fsdp"         , module = FSDPStrategy)
STRATEGIES.register(name = "hpu_parallel" , module = HPUParallelStrategy)
STRATEGIES.register(name = "hpu_single"   , module = SingleHPUStrategy)
STRATEGIES.register(name = "ipu"          , module = IPUStrategy)
STRATEGIES.register(name = "parallel"     , module = ParallelStrategy)
STRATEGIES.register(name = "single_device", module = SingleDeviceStrategy)
STRATEGIES.register(name = "single_tpu"   , module = SingleTPUStrategy)
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
        console.log(
            f"cuDNN available: [bright_green]True[/bright_green], "
            f"used:" + "[bright_green]True" if cudnn else "[red]False"
        )
    else:
        console.log(f"cuDNN available: [red]False")
    
    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        if platform.system() == "Windows":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
            console.log(
                f"Running on a Windows machine, set torch distributed backend "
                f"to gloo."
            )
        elif platform.system() == "Linux":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            console.log(
                f"Running on a Unix machine, set torch distributed backend to "
                f"nccl."
            )
            
# endregion
