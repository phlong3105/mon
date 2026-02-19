#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""System-wise Utilities.

This module provides helpers for terminal control and random seed management.
"""

from __future__ import annotations

__all__ = [
    "set_random_seed",
]

import os
import random
from typing import Sequence

import numpy as np
import torch

from .typing import IntOrTuple2


# ==============================================================================
# region CONTROL
# ==============================================================================

def set_random_seed(seed: IntOrTuple2, deterministic: bool = False):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed (IntOr2Tuple): Single seed value or a range of [min, max] from
            which a seed will be randomly sampled.
        deterministic (bool, optional): If True, configures PyTorch for
            deterministic behavior. Defaults to False.
    """
    if isinstance(seed, Sequence):
        # If a range is provided, sample a seed from it.
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]

    # Set seeds for all relevant libraries.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Configure PyTorch for deterministic behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms, warning if they are not available.
        torch.use_deterministic_algorithms(True, warn_only=True)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
