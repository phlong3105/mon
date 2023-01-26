#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements miscellaneous functions."""

from __future__ import annotations

__all__ = [
    "init_seeds", "to_size",
]

import random

import numpy as np
import torch
from torch.backends import cudnn

from mon.coreml.typing import Ints


def init_seeds(seed_value: int = 0):
    """Initialize the seeds of several backends (random, numpy, and torch) to
    ensure reproducibility.
    
    Args:
        seed_value: The random number generator.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    # Speed-reproducibility tradeoff:
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cudnn.is_available():
        if random.seed == 0:  # Slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark     = False
        else:                 # Faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark     = True


def to_size(size: Ints) -> tuple[int, int]:
    """Casts the size of arbitrary into [H, W] format.
    
    Args:
        size: Can be the size of the image, windows, kernels, etc.
    
    Returns:
        A tuple of the given size in [H, W] format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            size = size[1:3]
        if len(size) == 1:
            size = (size[0], size[0])
    elif isinstance(size, int):
        size = (size, size)
    return tuple(size)
