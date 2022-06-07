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
    "VibrateLR"
]


# MARK: - VibrateLR

@SCHEDULERS.register(name="vibrate_lr")
class VibrateLR(_LRScheduler):
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
    
        f = 0.1
        if process < 3 / 8:
            f = 1 - process * 8 / 3
        elif process < 5 / 8:
            f = 0.2
    
        T  = self.total_iter // 80
        Th = T // 2
        t  = self.last_epoch % T
    
        f2 = t / Th
        if t >= Th:
            f2 = 2 - f2
    
        weight = f * f2
    
        if self.last_epoch < Th:
            weight = max(0.1, weight)
    
        # print('f {}, T {}, Th {}, t {}, f2 {}'.format(f, T, Th, t, f2))
        return [weight * group["initial_lr"] for group in
                self.optimizer.param_groups]
