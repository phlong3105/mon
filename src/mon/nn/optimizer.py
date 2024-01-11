#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements optimizers and learning rate schedulers using the
:mod:`torch` package.
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
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LBFGS",
    "LRScheduler",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
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

from mon import core
from torch import optim
from torch.optim import lr_scheduler
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from mon.globals import LR_SCHEDULERS, OPTIMIZERS

console = core.console
math    = core.math


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


# region Learning Rate Scheduler

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
        restart_weights: Restart weights at each restart iteration. Default: ``[1]``.
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
        self.periods         = periods
        self.restart_weights = restart_weights
        self.eta_min         = eta_min
        assert (len(self.periods) == len(self.restart_weights)), \
            "``periods`` and ``restart_weights`` should have the same length."
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self.get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight  = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period  = self.periods[idx]
        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

    def get_position_from_periods(self, iteration: int, cumulative_period: list[int]) -> int:
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
LR_SCHEDULERS.register(name="multiStep_lr"                  , module=MultiStepLR)
LR_SCHEDULERS.register(name="multiplicative_lr"             , module=MultiplicativeLR)
LR_SCHEDULERS.register(name="one_cycle_lr"                  , module=OneCycleLR)
LR_SCHEDULERS.register(name="polynomial_lr"                 , module=PolynomialLR)
LR_SCHEDULERS.register(name="reduce_lr_on_plateau"          , module=ReduceLROnPlateau)
LR_SCHEDULERS.register(name="sequential_lr"                 , module=SequentialLR)
LR_SCHEDULERS.register(name="step_lr"                       , module=StepLR)

# endregion
