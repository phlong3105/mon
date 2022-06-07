#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
import platform
from typing import Union

import torch
from pytorch_lightning.plugins import DDP2Plugin
from pytorch_lightning.plugins import DDPPlugin
from torch import distributed as dist
from torch import nn

from one.core import Callable
from one.core import console

__all__ = [
	"get_dist_info",
	"is_parallel",
	"set_distributed_backend",
]


# MARK: - Functional

def get_dist_info():
	if dist.is_available():
		initialized = dist.is_initialized()
	else:
		initialized = False
	if initialized:
		rank       = dist.get_rank()
		world_size = dist.get_world_size()
	else:
		rank       = 0
		world_size = 1
	return rank, world_size


def is_parallel(model):
	return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def set_distributed_backend(strategy: Union[str, Callable], cudnn: bool = True):
	# NOTE: cuDNN
	if torch.backends.cudnn.is_available():
		torch.backends.cudnn.enabled = cudnn
		console.log(
			f"cuDNN available: [bright_green]True[/bright_green], "
			f"used:" + "[bright_green]True" if cudnn else "[red]False"
		)
	else:
		console.log(f"cuDNN available: [red]False")
	
	# NOTE: Torch Distributed Backend
	if strategy in ["ddp", "ddp2"] or isinstance(strategy, (DDPPlugin, DDP2Plugin)):
		if platform.system() == "Windows":
			os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
			console.log(
				"Running on a Windows machine, set torch distributed backend "
				"to gloo."
			)
		elif platform.system() == "Linux":
			os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
			console.log(
				"Running on a Unix machine, set torch distributed backend "
				"to nccl."
			)
