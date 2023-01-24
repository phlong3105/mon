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

import torch
import torch.cuda
from lightning.pytorch import accelerators, strategies
from torch import distributed

from mon import core
from mon.coreml import constant
from mon.coreml.typing import CallableType

# region Accelerator

CPUAccelerator  = accelerators.CPUAccelerator
CUDAAccelerator = accelerators.CUDAAccelerator
HPUAccelerator  = accelerators.HPUAccelerator
IPUAccelerator  = accelerators.IPUAccelerator
MPSAccelerator  = accelerators.MPSAccelerator
TPUAccelerator  = accelerators.TPUAccelerator

constant.ACCELERATOR.register(name="cpu",  module=CPUAccelerator)
constant.ACCELERATOR.register(name="cuda", module=CUDAAccelerator)
constant.ACCELERATOR.register(name="gpu",  module=CUDAAccelerator)
constant.ACCELERATOR.register(name="hpu",  module=HPUAccelerator)
constant.ACCELERATOR.register(name="ipu",  module=IPUAccelerator)
constant.ACCELERATOR.register(name="mps",  module=MPSAccelerator)
constant.ACCELERATOR.register(name="tpu",  module=TPUAccelerator)

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

constant.STRATEGY.register(name="bagua",             module=BaguaStrategy)
constant.STRATEGY.register(name="collaborative",     module=HivemindStrategy)
constant.STRATEGY.register(name="colossalai",        module=ColossalAIStrategy)
constant.STRATEGY.register(name="fsdp_native",       module=DDPFullyShardedNativeStrategy)
constant.STRATEGY.register(name="fsdp",              module=DDPFullyShardedStrategy)
constant.STRATEGY.register(name="ddp_sharded",       module=DDPShardedStrategy)
constant.STRATEGY.register(name="ddp_sharded_spawn", module=DDPSpawnShardedStrategy)
constant.STRATEGY.register(name="ddp_spawn",         module=DDPSpawnStrategy)
constant.STRATEGY.register(name="ddp",               module=DDPStrategy)
constant.STRATEGY.register(name="dp",                module=DataParallelStrategy)
constant.STRATEGY.register(name="deepspeed",         module=DeepSpeedStrategy)
constant.STRATEGY.register(name="horovod",           module=HorovodStrategy)
constant.STRATEGY.register(name="hpu_parallel",      module=HPUParallelStrategy)
constant.STRATEGY.register(name="hpu_single",        module=SingleHPUStrategy)
constant.STRATEGY.register(name="ipu_strategy",      module=IPUStrategy)
constant.STRATEGY.register(name="tpu_spawn",         module=TPUSpawnStrategy)
constant.STRATEGY.register(name="single_tpu",        module=SingleTPUStrategy)

# endregion


# region Helper Function

def get_distributed_info() -> tuple[int, int]:
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
    return rank, world_size


def set_distributed_backend(strategy: str | CallableType, cudnn: bool = True):
    """If you're running on Windows, set the distributed backend to gloo. If
    you're running on Linux, set the distributed backend to nccl.
    
    Args:
        strategy: The distributed strategy to use. One of ["ddp", "ddp2"].
        cudnn: Whether to use cuDNN or not. Defaults to True.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        core.console.log(
            f"cuDNN available: [bright_green]True[/bright_green], "
            f"used:" + "[bright_green]True" if cudnn else "[red]False"
        )
    else:
        core.console.log(f"cuDNN available: [red]False")
    
    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        if platform.system() == "Windows":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
            core.console.log(
                "Running on a Windows machine, set torch distributed backend "
                "to gloo."
            )
        elif platform.system() == "Linux":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            core.console.log(
                "Running on a Unix machine, set torch distributed backend "
                "to nccl."
            )
            
# endregion
