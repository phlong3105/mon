#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from one.core import MODULE_WRAPPERS

__all__ = [
    "is_module_wrapper",
]


# MARK: - Builder

MODULE_WRAPPERS.register(name="dp",  module=DataParallel)
MODULE_WRAPPERS.register(name="ddp", module=DistributedDataParallel)


# MARK: - Functional

def is_module_wrapper(module: nn.Module):
    """Check if a module is a module wrapper. Following 3 modules (and
    their subclasses) are regarded as module wrappers: DataParallel,
    DistributedDataParallel. You may add you own module wrapper by registering
    it to MODULE_WRAPPERS.
    """
    module_wrappers = tuple(MODULE_WRAPPERS.registry.values())
    return isinstance(module, module_wrappers)
