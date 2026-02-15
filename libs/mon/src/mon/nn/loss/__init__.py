#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimizers & Schedulers.

This package contains optimizers and learning rate schedulers for training
neural networks.

File Structure:
::

    loss/
    ├── __init__.py
    ├── base.py          # Base classes and mixins
    ├── common.py        # Common losses (e.g., Cross-Entropy, MSE, SmoothL1)
    ├── generative.py    # Generative model losses (e.g., KL-Divergence, GAN, VAE)
    └── vision.py        # Image and video losses (e.g., IoU, Focal Loss, Perceptual Loss)
"""

from __future__ import annotations

from .base import *
from .common import *
from .generative import *
from .vision import *
