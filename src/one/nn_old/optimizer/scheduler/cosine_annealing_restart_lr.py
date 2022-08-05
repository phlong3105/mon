#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cosine annealing with restarts learning rate scheme.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from one.core import SCHEDULERS

__all__ = [
    "CosineAnnealingRestartLR",
    "get_position_from_periods"
]


# MARK: - CosineAnnealingRestartLR

def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int):
            Current iteration.
        cumulative_period (list[int]):
            Cumulative period list.

    Returns:
        int: Position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
        

@SCHEDULERS.register(name="cosine_annealing_restart_lr")
class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        periods (list):
            Period for each cosine anneling cycle.
        restart_weights (list):
            Restart weights at each restart iteration. Default: `[1]`.
        eta_min (float):
            Minimum lr. Default: `0`.
        last_epoch (int):
            Used in _LRScheduler. Default: `-1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        optimizer      : Optimizer,
        periods        : list,
        restart_weights: list = (1, ),
        eta_min        : int  = 0,
        last_epoch     : int  = -1
    ):
        self.periods         = periods
        self.restart_weights = restart_weights
        self.eta_min         = eta_min
        if len(self.periods) != len(self.restart_weights):
            raise ValueError(f"`periods` and `restart_weights` must have the same length."
                             f" But got: {len(self.periods)} != {len(self.restart_weights)}.")
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super().__init__(optimizer, last_epoch)
    
    # MARK: Forward Pass
    
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight  = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period  = self.periods[idx]
    
        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) /
                                     current_period)))
            for base_lr in self.base_lrs
        ]
