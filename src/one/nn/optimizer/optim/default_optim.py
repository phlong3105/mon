#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torch import optim

from one.core import OPTIMIZERS

# MARK: - Register

OPTIMIZERS.register(name="adadelta",    module=optim.Adadelta)
OPTIMIZERS.register(name="adagrad",     module=optim.Adagrad)
OPTIMIZERS.register(name="adam",        module=optim.Adam)
OPTIMIZERS.register(name="adamax",      module=optim.Adamax)
OPTIMIZERS.register(name="adam_w",      module=optim.AdamW)
OPTIMIZERS.register(name="adamw",       module=optim.AdamW)
OPTIMIZERS.register(name="asgd",        module=optim.ASGD)
OPTIMIZERS.register(name="lbfgs",       module=optim.LBFGS)
OPTIMIZERS.register(name="rms_prop",    module=optim.RMSprop)
OPTIMIZERS.register(name="r_prop",      module=optim.Rprop)
OPTIMIZERS.register(name="sgd",         module=optim.SGD)
OPTIMIZERS.register(name="sparse_adam", module=optim.SparseAdam)
