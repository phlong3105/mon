#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements optimizers and learning rate schedulers using the
:obj:`torch` package.
"""

from __future__ import annotations

__all__ = [
    "ASGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingRestartCyclicLR",
    "CosineAnnealingRestartLR",
    "CosineAnnealingRestartLR2",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "GradualWarmupScheduler",
    "LBFGS",
    "LRScheduler",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "MultiStepLRRestart",
    "MultiplicativeLR",
    "NAdam",
    "OneCycleLR",
    "Optimizer",
    "PolynomialLR",
    "RAdam",
    "RMSprop",
    "ReduceLROnPlateau",
    "Rprop",
    "SGD",
    "SequentialLR",
    "SparseAdam",
    "StepLR",
]

import math
from collections import Counter, defaultdict
from typing import Any

from torch import optim
from torch.optim import lr_scheduler
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from mon.globals import LR_SCHEDULERS, OPTIMIZERS

# region Optimizer

Optimizer       = optim.Optimizer
Adadelta        = optim.Adadelta
Adagrad         = optim.Adagrad
Adam            = optim.Adam
Adamax          = optim.Adamax
AdamW           = optim.AdamW
ASGD            = optim.ASGD
LBFGS           = optim.LBFGS
NAdam           = optim.NAdam
RAdam           = optim.RAdam
RMSprop         = optim.RMSprop
Rprop           = optim.Rprop
SGD             = optim.SGD
SparseAdam      = optim.SparseAdam

OPTIMIZERS.register(name="adadelta"   , module=Adadelta)
OPTIMIZERS.register(name="adagrad"    , module=Adagrad)
OPTIMIZERS.register(name="adam"       , module=Adam)
OPTIMIZERS.register(name="adamax"     , module=Adamax)
OPTIMIZERS.register(name="adamw"      , module=AdamW)
OPTIMIZERS.register(name="asgd"       , module=ASGD)
OPTIMIZERS.register(name="lbfgs"      , module=LBFGS)
OPTIMIZERS.register(name="nadam"      , module=NAdam)
OPTIMIZERS.register(name="radam"      , module=RAdam)
OPTIMIZERS.register(name="rmsprop"    , module=RMSprop)
OPTIMIZERS.register(name="rprop"      , module=Rprop)
OPTIMIZERS.register(name="sgd"        , module=SGD)
OPTIMIZERS.register(name="sparse_adam", module=SparseAdam)

# endregion


# region Scheduler

@LR_SCHEDULERS.register(name="cosine_annealing_restart_lr")
class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods         = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min         = 1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer: Torch optimizer.
        periods: Period for each cosine annealing cycle.
        restart_weights: Restart weights at each restart iteration.
            Default: ``[1]``.
        eta_min: The minimum lr. Default: ``0``.
        last_epoch: Used in _LRScheduler. Default: ``-1``.
    """

    def __init__(
        self,
        optimizer      : Optimizer,
        periods        : list | tuple[int, ...],
        restart_weights: list | tuple[int, ...] = (1, ),
        eta_min        : int = 0,
        last_epoch     : int = -1
    ):
        self.periods           = periods
        self.restart_weights   = restart_weights
        self.eta_min           = eta_min
        self.cumulative_period = [sum(self.periods[0:i + 1])
                                  for i in range(0, len(self.periods))]
        if len(self.periods) != len(self.restart_weights):
            raise ValueError(f"`periods` and `restart_weights` should have the "
                             f"same length.")
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self._get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight  = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period  = self.periods[idx]
        return [
            self.eta_min
            + current_weight * 0.5 * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

    @staticmethod
    def _get_position_from_periods(
        iteration        : int,
        cumulative_period: list[int]
    ) -> int:
        """Get the position from a period list.

        It will return the index of the right-closest number in the period list.
        For example, the cumulative_period = [100, 200, 300, 400],
        if iteration == 50,  return 0;
        if iteration == 210, return 2;
        if iteration == 300, return 2.

        Args:
            iteration: Current iteration.
            cumulative_period: Cumulative period list.

        Returns:
            The position of the right-closest number in the period list.
        """
        for i, period in enumerate(cumulative_period):
            if iteration <= period:
                return i


@LR_SCHEDULERS.register(name="cosine_annealing_restart_lr2")
class CosineAnnealingRestartLR2(_LRScheduler):
    
    def __init__(
        self,
        optimizer      : Optimizer,
        periods        : list | tuple[int, ...],
        restarts       : list | tuple[int, ...],
        restart_weights: list | tuple[int, ...] = (1, ),
        eta_min        : int = 0,
        last_epoch     : int = -1
    ):
        self.periods         = periods
        self.t_max           = self.periods[0]  # current T period
        self.eta_min         = eta_min
        self.restarts        = restarts if restarts else [0]
        self.restarts        = [v + 1 for v in self.restarts]
        self.restart_weights = restart_weights if restart_weights else [1]
        self.last_restart    = 0
        
        if len(self.restarts) != len(self.restart_weights):
            raise ValueError(f"`restarts` and `restart_weights` must have the "
                             f"same length.")
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.t_max        = self.periods[self.restarts.index(self.last_epoch) + 1]
            weight            = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.t_max) % (2 * self.t_max) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.t_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.t_max)) /
            (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.t_max)) *
            (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]


@LR_SCHEDULERS.register(name="cosine_annealing_restart_cyclic_lr")
class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing with restarts learning rate scheme. It has four cycles,
    each has 10 iterations. At 10th, 20th, 30th, the scheduler will restart with
    the weights in restart_weights.
    
    An example of config:
    periods         = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min         = 1e-7
    
    Args:
        optimizer: Torch optimizer.
        periods: Period for each cosine anneling cycle.
        restart_weights: Restart weights at each restart iteration.
            Default: ``[1]``.
        eta_min: The mimimum lr. Default: ``0``.
        last_epoch: Used in _LRScheduler. Default: ``-1``.
    """
    
    def __init__(
        self,
        optimizer      : Optimizer,
        periods        : list | tuple[int, ...],
        restart_weights: list | tuple[int, ...] = (1, ),
        eta_mins       : list | tuple[int, ...] = (0, ),
        last_epoch     : int                    = -1,
    ):
        self.periods           = periods
        self.restart_weights   = restart_weights
        self.eta_mins          = eta_mins
        self.cumulative_period = [sum(self.periods[0:i + 1])
                                  for i in range(0, len(self.periods))]
        
        if len(self.periods) != len(self.restart_weights):
            raise ValueError(f"`periods` and `restart_weights` must have the "
                             f"same length.")
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        idx = self._get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight  = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period  = self.periods[idx]
        eta_min         = self.eta_mins[idx]
        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
    
    def _get_position_from_periods(self, iteration: int, cumulative_period: list[int]) -> int:
        """Get the position from a period list.

        It will return the index of the right-closest number in the period list.
        For example, the cumulative_period = [100, 200, 300, 400],
        if iteration == 50,  return 0;
        if iteration == 210, return 2;
        if iteration == 300, return 2.

        Args:
            iteration: Current iteration.
            cumulative_period: Cumulative period list.

        Returns:
            The position of the right-closest number in the period list.
        """
        for i, period in enumerate(cumulative_period):
            if iteration <= period:
                return i


@LR_SCHEDULERS.register(name="gradual_warmup_scheduler")
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in an optimizer.
    
    Paper: `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`__

    Args:
        optimizer: A wrapped optimizer.
        multiplier: `Target learning rate = base lr * multiplier` if `multiplier > 1.0`.
            If `multiplier = 1.0`, lr starts from 0 and ends up with the base_lr.
        total_epoch: Target learning rate is reached at total_epoch.
        after_scheduler: After target_epoch, use this scheduler (e.g., ReduceLROnPlateau)
    """
    
    def __init__(
        self,
        optimizer      : Optimizer,
        multiplier     : int,
        total_epoch    : int,
        after_scheduler: _LRScheduler | None = None
    ):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("`multiplier` should be greater thant or equal to 1.0")
        self.total_epoch     = total_epoch
        self.after_scheduler = after_scheduler
        self.finished        = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]
    
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
    
    def step(self, epoch=None, metrics=None):
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


@LR_SCHEDULERS.register(name="multistep_lr_restart")
class MultiStepLRRestart(_LRScheduler):
    
    def __init__(
        self,
        optimizer  : Optimizer,
        milestones : Any,
        restarts   : list | tuple[int, ...] | None = None,
        weights    : list | tuple[int, ...] | None = None,
        gamma      : float = 0.1,
        clear_state: bool  = False,
        last_epoch : int   = -1
    ):
        self.milestones      = Counter(milestones)
        self.gamma           = gamma
        self.clear_state     = clear_state
        self.restarts        = restarts if restarts else [0]
        self.restarts        = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(self.restart_weights), f"Restarts, and their weights do not match."
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


@LR_SCHEDULERS.register(name="vibrate_lr")
class VibrateLR(_LRScheduler):
    """

    Args:
        optimizer: Torch optimizer.
        last_epoch: Used in _LRScheduler. Default: ``-1``.
    """

    def __init__(
        self,
        optimizer : Optimizer,
        total_iter: int,
        last_epoch: int = -1,
    ):
        self.total_iter = total_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        process = self.last_epoch / self.total_iter

        f = 0.1
        if process < 3 / 8:
            f = 1 - process * 8 / 3
        elif process < 5 / 8:
            f = 0.2

        t  = self.total_iter // 80
        th = t // 2

        t  = self.last_epoch % t

        f2 = t / th
        if t >= th:
            f2 = 2 - f2

        weight = f * f2

        if self.last_epoch < th:
            weight = max(0.1, weight)

        # console.log('f {}, t {}, th {}, t {}, f2 {}'.format(f, t, th, t, f2))
        return [weight * group["initial_lr"] for group in self.optimizer.param_groups]


LRScheduler                 = _LRScheduler
ChainedScheduler            = lr_scheduler.ChainedScheduler
ConstantLR                  = lr_scheduler.ConstantLR
CosineAnnealingLR           = lr_scheduler.CosineAnnealingLR
CosineAnnealingWarmRestarts = lr_scheduler.CosineAnnealingWarmRestarts
CyclicLR                    = lr_scheduler.CyclicLR
ExponentialLR               = lr_scheduler.ExponentialLR
LambdaLR                    = lr_scheduler.LambdaLR
LinearLR                    = lr_scheduler.LinearLR
MultiStepLR                 = lr_scheduler.MultiStepLR
MultiplicativeLR            = lr_scheduler.MultiplicativeLR
OneCycleLR                  = lr_scheduler.OneCycleLR
PolynomialLR                = lr_scheduler.PolynomialLR
ReduceLROnPlateau           = lr_scheduler.ReduceLROnPlateau
SequentialLR                = lr_scheduler.SequentialLR
StepLR                      = lr_scheduler.StepLR

LR_SCHEDULERS.register(name="chained_scheduler"             , module=ChainedScheduler)
LR_SCHEDULERS.register(name="constant_lr"                   , module=ConstantLR)
LR_SCHEDULERS.register(name="cosine_annealing_lr"           , module=CosineAnnealingLR)
LR_SCHEDULERS.register(name="cosine_annealing_warm_restarts", module=CosineAnnealingWarmRestarts)
LR_SCHEDULERS.register(name="cyclic_lr"                     , module=CyclicLR)
LR_SCHEDULERS.register(name="exponential_lr"                , module=ExponentialLR)
LR_SCHEDULERS.register(name="lambda_lr"                     , module=LambdaLR)
LR_SCHEDULERS.register(name="linear_lr"                     , module=LinearLR)
LR_SCHEDULERS.register(name="multistep_lr"                  , module=MultiStepLR)
LR_SCHEDULERS.register(name="multiplicative_lr"             , module=MultiplicativeLR)
LR_SCHEDULERS.register(name="one_cycle_lr"                  , module=OneCycleLR)
LR_SCHEDULERS.register(name="polynomial_lr"                 , module=PolynomialLR)
LR_SCHEDULERS.register(name="reduce_lr_on_plateau"          , module=ReduceLROnPlateau)
LR_SCHEDULERS.register(name="sequential_lr"                 , module=SequentialLR)
LR_SCHEDULERS.register(name="step_lr"                       , module=StepLR)

# endregion
