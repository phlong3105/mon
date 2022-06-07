#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from one.core import SCHEDULERS

# MARK: - Register

SCHEDULERS.register(name="cosine_annealing_lr",            module=CosineAnnealingLR)
SCHEDULERS.register(name="cosine_annealing_warm_restarts", module=CosineAnnealingWarmRestarts)
SCHEDULERS.register(name="cyclic_lr",                      module=CyclicLR)
SCHEDULERS.register(name="exponential_lr",                 module=ExponentialLR)
SCHEDULERS.register(name="lambda_lr",                      module=LambdaLR)
SCHEDULERS.register(name="multistep_lr",                   module=MultiStepLR)
SCHEDULERS.register(name="reduce_lr_on_plateau",           module=ReduceLROnPlateau)
SCHEDULERS.register(name="step_lr",                        module=StepLR)
