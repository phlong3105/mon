#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimizers and Schedulers
"""

from __future__ import annotations

import math

from torch.optim import *
from torch.optim.lr_scheduler import *
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from one.constants import OPTIMIZERS
from one.constants import SCHEDULERS


# H1: - Optimizer --------------------------------------------------------------

OPTIMIZERS.register(name="adadelta",    module=Adadelta)
OPTIMIZERS.register(name="adagrad",     module=Adagrad)
OPTIMIZERS.register(name="adam",        module=Adam)
OPTIMIZERS.register(name="adamax",      module=Adamax)
OPTIMIZERS.register(name="adam_w",      module=AdamW)
OPTIMIZERS.register(name="asgd",        module=ASGD)
OPTIMIZERS.register(name="lbfgs",       module=LBFGS)
OPTIMIZERS.register(name="n_adam",      module=NAdam)
OPTIMIZERS.register(name="r_adam",      module=RAdam)
OPTIMIZERS.register(name="rms_prop",    module=RMSprop)
OPTIMIZERS.register(name="r_prop",      module=Rprop)
OPTIMIZERS.register(name="sgd",         module=SGD)
OPTIMIZERS.register(name="sparse_adam", module=SparseAdam)


# H1: - Scheduler --------------------------------------------------------------

SCHEDULERS.register(name="chained_scheduler",              module=ChainedScheduler)
SCHEDULERS.register(name="constant_lr",                    module=ConstantLR)
SCHEDULERS.register(name="cosine_annealing_lr",            module=CosineAnnealingLR)
SCHEDULERS.register(name="cosine_annealing_warm_restarts", module=CosineAnnealingWarmRestarts)
SCHEDULERS.register(name="cyclic_lr",                      module=CyclicLR)
SCHEDULERS.register(name="exponential_lr",                 module=ExponentialLR)
SCHEDULERS.register(name="lambda_lr",                      module=LambdaLR)
SCHEDULERS.register(name="linear_lr",                      module=LinearLR)
SCHEDULERS.register(name="multistep_lr",                   module=MultiStepLR)
SCHEDULERS.register(name="reduce_lr_on_plateau",           module=ReduceLROnPlateau)
SCHEDULERS.register(name="sequential_lr",                  module=SequentialLR)
SCHEDULERS.register(name="step_lr",                        module=StepLR)
