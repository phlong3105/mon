#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""System-wise utilities.

This module provides helpers for terminal control and for setting reproducible
seeds across Python, NumPy, and PyTorch to enable consistent terminal clearing
and experiment reproducibility in the codebase.
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

def set_random_seed(seed: int | tuple[int, int], deterministic: bool = False):
    """Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: A single integer seed or a two-element tuple `(min, max)` from
            which a seed will be randomly sampled.
        deterministic: If True, configures PyTorch for deterministic
            operations, which may have a performance cost.
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
    """Clear the terminal screen using platform-specific commands.

    This function uses ANSI escape codes for POSIX systems (Linux, macOS) and
    the `cls` command for Windows, providing a more efficient and secure
    alternative to `os.system("clear")`.
    """
    if platform.system() == "Windows":
        # For Windows, 'cls' is the standard command.
        os.system("cls")
    else:
        # For POSIX systems, use ANSI escape codes for efficiency.
        # \033[H moves the cursor to the top-left corner.
        # \033[2J clears the entire screen.
        print("\033[H\033[2J", end="", flush=True)

# endregion
