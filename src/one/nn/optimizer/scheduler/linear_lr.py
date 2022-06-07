#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from one.core import SCHEDULERS

__all__ = [
    "LinearLR"
]


# MARK: - LinearLR

@SCHEDULERS.register(name="linear_lr")
class LinearLR(_LRScheduler):
    """

    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        last_epoch (int):
            Used in _LRScheduler. Default: `-1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, optimizer: Optimizer, total_iter, last_epoch: int = -1):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)
    
    # MARK: Forward Pass
    
    def get_lr(self):
        process = self.last_epoch / self.total_iter
        weight  = (1 - process)
        return [weight * group["initial_lr"] for group in
                self.optimizer.param_groups]
