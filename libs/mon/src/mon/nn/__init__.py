#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural Networks.

This package contains the building blocks for constructing neural networks.

File Structure:
::

    mon.nn/
    ├── __init__.py
    ├── loss/           # Loss functions
    ├── models/         # Meta-architectures used across domains
    ├── modules/        # Atomic components
    └── optim/          # Optimizers & Schedulers
"""

from __future__ import annotations

from .loss import *
from .models import *
from .modules import *
from .optim import *
