#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One math library.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch.backends import cudnn


# H1: - Random ---------------------------------------------------------------

def init_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Speed-reproducibility tradeoff
    # https://pytorch.org/docs/stable/notes/randomness.html
    if cudnn.is_available():
        if seed == 0:  # slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark     = False
        else:  # faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark     = True


def make_divisible(x: Any, divisor: int):
    """
    Returns x evenly divisible by divisor.
    """
    return math.ceil(x / divisor) * divisor
