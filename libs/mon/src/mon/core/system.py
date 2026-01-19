#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""System-wise utilities.

This module provides helpers for terminal control and random seed management.
"""

from __future__ import annotations

__all__ = [
    "clear_terminal",
    "set_random_seed",
]

import os
import platform
import random
from typing import Sequence

import numpy as np
import torch


# ==============================================================================
# region CONTROL
# ==============================================================================

def set_random_seed(
    seed         : int | tuple[int, int],
    deterministic: bool = False,
):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Single integer seed or a two-element tuple (min, max) from which a
            value for ``seed`` will be randomly sampled.
        deterministic: If ``deterministic`` is True, configure PyTorch for
            deterministic operations. Defaults to False.
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
        torch.backends.cudnn.benchmark     = False
        # Use deterministic algorithms, warning if they are not available.
        torch.use_deterministic_algorithms(True, warn_only=True)

# endregion


# ==============================================================================
# region BASIC LOGGING
# ==============================================================================

def clear_terminal():
    """Clear the terminal screen."""
    if platform.system() == "Windows":
        # For Windows, 'cls' is the standard command.
        os.system("cls")
    else:
        # For POSIX systems, use ANSI escape codes for efficiency.
        # \033[H moves the cursor to the top-left corner.
        # \033[2J clears the entire screen.
        print("\033[H\033[2J", end="", flush=True)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
