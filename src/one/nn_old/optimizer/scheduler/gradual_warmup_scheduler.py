#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gradually warm-up(increasing) learning rate in optimizer.
"""

from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from one.core import SCHEDULERS

__all__ = [
    "GradualWarmupScheduler"
]


# MARK: - GradualWarmupScheduler

@SCHEDULERS.register(name="gradual_warmup_scheduler")
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer. Proposed in
    'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        multiplier (float):
            Target learning rate = base lr * multiplier if  multiplier > 1.0.
            If multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch (int):
            Target learning rate is reached at total_epoch, gradually.
        after_scheduler (Optimizer):
            After target_epoch, use this scheduler(eg. ReduceLROnPlateau).
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        optimizer      : Optimizer,
        multiplier     : float,
        total_epoch    : int,
        after_scheduler: Optimizer = None
    ):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError(f"`multiplier` must >= 1.0. But got: {self.multiplier}.")
        self.total_epoch     = total_epoch
        self.after_scheduler = after_scheduler
        self.finished        = False
        super().__init__(optimizer)
    
    # MARK: Forward Pass
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch)
                    for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.) * self.last_epoch /
                           self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch /
                           self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch: Optional[int] = None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
