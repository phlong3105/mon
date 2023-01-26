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

from mon.coreml import constant

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

constant.OPTIMIZER.register(name="adadelta",    module=Adadelta)
constant.OPTIMIZER.register(name="adagrad",     module=Adagrad)
constant.OPTIMIZER.register(name="adam",        module=Adam)
constant.OPTIMIZER.register(name="adamax",      module=Adamax)
constant.OPTIMIZER.register(name="adamw",       module=AdamW)
constant.OPTIMIZER.register(name="asgd",        module=ASGD)
constant.OPTIMIZER.register(name="lbfgs",       module=LBFGS)
constant.OPTIMIZER.register(name="nadam",       module=NAdam)
constant.OPTIMIZER.register(name="radam",       module=RAdam)
constant.OPTIMIZER.register(name="rmsprop",     module=RMSprop)
constant.OPTIMIZER.register(name="rprop",       module=Rprop)
constant.OPTIMIZER.register(name="sgd",         module=SGD)
constant.OPTIMIZER.register(name="sparse_adam", module=SparseAdam)

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

constant.LR_SCHEDULER.register(name="chained_scheduler",              module=ChainedScheduler)
constant.LR_SCHEDULER.register(name="constant_lr",                    module=ConstantLR)
constant.LR_SCHEDULER.register(name="cosine_annealing_lr",            module=CosineAnnealingLR)
constant.LR_SCHEDULER.register(name="cosine_annealing_warm_restarts", module=CosineAnnealingWarmRestarts)
constant.LR_SCHEDULER.register(name="cyclic_lr",                      module=CyclicLR)
constant.LR_SCHEDULER.register(name="exponential_lr",                 module=ExponentialLR)
constant.LR_SCHEDULER.register(name="lambda_lr",                      module=LambdaLR)
constant.LR_SCHEDULER.register(name="linear_lr",                      module=LinearLR)
constant.LR_SCHEDULER.register(name="multiStep_lr",                   module=MultiStepLR)
constant.LR_SCHEDULER.register(name="reduce_lr_on_plateau",           module=ReduceLROnPlateau)
constant.LR_SCHEDULER.register(name="sequential_lr",                  module=SequentialLR)
constant.LR_SCHEDULER.register(name="step_lr",                        module=StepLR)

# endregion
