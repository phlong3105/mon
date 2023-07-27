#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements optimizers and learning rate schedulers using the
:mod:`torch` package.
"""

from __future__ import annotations

__all__ = [
    "ASGD", "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax",
    "ChainedScheduler", "ConstantLR", "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts", "CyclicLR", "ExponentialLR", "LBFGS",
    "LRScheduler", "LambdaLR", "LinearLR", "MultiStepLR", "NAdam", "Optimizer",
    "RAdam", "RMSprop", "ReduceLROnPlateau", "Rprop", "SGD", "SequentialLR",
    "SparseAdam", "StepLR",
]

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


# region Learning Rate Scheduler

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
LR_SCHEDULERS.register(name="reduce_lr_on_plateau"          , module=ReduceLROnPlateau)
LR_SCHEDULERS.register(name="sequential_lr"                 , module=SequentialLR)
LR_SCHEDULERS.register(name="step_lr"                       , module=StepLR)

# endregion
