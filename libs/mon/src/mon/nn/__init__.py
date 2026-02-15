#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural Network.

This package contains the building blocks for constructing neural networks.

File Structure:
::

    mon.nn/
    ├── __init__.py
    ├── loss/               # Loss functions
    │   ├── base.py
    │   ├── common.py       # Common losses (e.g., Cross-Entropy, MSE, SmoothL1)
    │   ├── generative.py   # Generative model losses (e.g., KL-Divergence, GAN, VAE)
    │   └── vision.py       # Image and video losses (e.g., IoU, Focal Loss, Perceptual Loss)
    ├── models/             # Meta-architectures used across domains
    │   ├── backbone/       # Feature extractors
    │   ├── head/           # Output heads
    │   └── neck/           # Feature aggregators
    ├── modules/            # Atomic components
    └── optim/              # Optimizers & Schedulers
        ├── optimizer.py    # Optimizers
        ├── scheduler.py    # Learning rate schedulers
        └── ema.py          # Exponential moving average
"""

from __future__ import annotations
