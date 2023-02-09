#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements strategies used during training machine learning
models. A strategy is a composition of one Accelerator, one Precision Plugin, a
CheckpointIO plugin, and other optional plugins such as the ClusterEnvironment.

References:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
"""

from __future__ import annotations

__all__ = [
    "BaguaStrategy", "CPUAccelerator", "CUDAAccelerator", "ColossalAIStrategy",
    "DDPFullyShardedNativeStrategy", "DDPFullyShardedStrategy",
    "DDPShardedStrategy", "DDPSpawnShardedStrategy", "DDPSpawnStrategy",
    "DDPStrategy", "DataParallelStrategy", "DeepSpeedStrategy",
    "HPUAccelerator", "HPUParallelStrategy", "HivemindStrategy",
    "HorovodStrategy", "IPUAccelerator", "IPUStrategy", "MPSAccelerator",
    "SingleHPUStrategy", "SingleTPUStrategy", "TPUAccelerator",
    "TPUSpawnStrategy",
]

import os
import platform
from typing import Callable

import torch
import torch.cuda
from lightning.pytorch import accelerators, strategies
from torch import distributed

from mon.foundation import console
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

BaguaStrategy                 = strategies.BaguaStrategy
HivemindStrategy              = strategies.HivemindStrategy
ColossalAIStrategy            = strategies.ColossalAIStrategy
DDPFullyShardedNativeStrategy = strategies.DDPFullyShardedNativeStrategy
DDPFullyShardedStrategy       = strategies.DDPFullyShardedStrategy
DDPShardedStrategy            = strategies.DDPShardedStrategy
DDPSpawnShardedStrategy       = strategies.DDPSpawnShardedStrategy
DDPSpawnStrategy              = strategies.DDPSpawnStrategy
DDPStrategy                   = strategies.DDPStrategy
DataParallelStrategy          = strategies.DataParallelStrategy
DeepSpeedStrategy             = strategies.DeepSpeedStrategy
HorovodStrategy               = strategies.HorovodStrategy
HPUParallelStrategy           = strategies.HPUParallelStrategy
SingleHPUStrategy             = strategies.SingleHPUStrategy
IPUStrategy                   = strategies.IPUStrategy
TPUSpawnStrategy              = strategies.TPUSpawnStrategy
SingleTPUStrategy             = strategies.SingleTPUStrategy

STRATEGIES.register(name="bagua"            , module=BaguaStrategy)
STRATEGIES.register(name="collaborative"    , module=HivemindStrategy)
STRATEGIES.register(name="colossalai"       , module=ColossalAIStrategy)
STRATEGIES.register(name="fsdp_native"      , module=DDPFullyShardedNativeStrategy)
STRATEGIES.register(name="fsdp"             , module=DDPFullyShardedStrategy)
STRATEGIES.register(name="ddp_sharded"      , module=DDPShardedStrategy)
STRATEGIES.register(name="ddp_sharded_spawn", module=DDPSpawnShardedStrategy)
STRATEGIES.register(name="ddp_spawn"        , module=DDPSpawnStrategy)
STRATEGIES.register(name="ddp"              , module=DDPStrategy)
STRATEGIES.register(name="dp"               , module=DataParallelStrategy)
STRATEGIES.register(name="deepspeed"        , module=DeepSpeedStrategy)
STRATEGIES.register(name="horovod"          , module=HorovodStrategy)
STRATEGIES.register(name="hpu_parallel"     , module=HPUParallelStrategy)
STRATEGIES.register(name="hpu_single"       , module=SingleHPUStrategy)
STRATEGIES.register(name="ipu_strategy"     , module=IPUStrategy)
STRATEGIES.register(name="tpu_spawn"        , module=TPUSpawnStrategy)
STRATEGIES.register(name="single_tpu"       , module=SingleTPUStrategy)

# endregion


# region Helper Function

def get_distributed_info() -> list[int]:
    """If distributed is available, return the rank and world size, otherwise
    return 0 and 1
    
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
    you're running on Linux, set the distributed backend to nccl.
    
    Args:
        strategy: The distributed strategy to use. One of ["ddp", "ddp2"].
        cudnn: Whether to use cuDNN or not. Defaults to True.
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
