#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MultiStep with restarts learning rate scheme.
"""

from __future__ import annotations

from collections import Counter

from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from one.core import SCHEDULERS

__all__ = [
    "MultiStepRestartLR"
]


# MARK: - MultiStepRestartLR

@SCHEDULERS.register(name="multistep_restart_lr")
class MultiStepRestartLR(_LRScheduler):
    """MultiStep with restarts learning rate scheme.

    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        milestones (list):
            Iterations that will decrease learning rate.
        gamma (float):
            Decrease ratio. Default: `0.1`.
        restarts (list):
            Restart iterations. Default: `[0]`.
        restart_weights (list):
            Restart weights at each restart iteration. Default: `[1]`.
        last_epoch (int):
            Used in _LRScheduler. Default: `-1`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        optimizer      : Optimizer,
        milestones     : list,
        gamma          : float = 0.1,
        restarts       : list  = (0, ),
        restart_weights: list  = (1, ),
        last_epoch     : int   = -1
    ):
        self.milestones      = Counter(milestones)
        self.gamma           = gamma
        self.restarts        = restarts
        self.restart_weights = restart_weights
        if len(self.restarts) != len(self.restart_weights):
            raise ValueError(
                f"`restarts` and `restart_weights` must have the same length. "
                f"But got: {len(self.restarts)} != {len(self.restart_weights)}."
            )
            
        super().__init__(optimizer, last_epoch)
    
    # MARK: Forward Pass
    
    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight
                    for group in self.optimizer.param_groups]
        
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]
