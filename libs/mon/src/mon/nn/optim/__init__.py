#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimizers & Schedulers.

This package contains optimizers and learning rate schedulers for training
neural networks.

File Structure:
::

    optim/
    ├── __init__.py
    ├── optimizer.py    # Optimizers
    ├── scheduler.py    # Learning rate schedulers
    └── ema.py          # Exponential moving average
"""

from __future__ import annotations

from .optimizer import *
from .scheduler import *
