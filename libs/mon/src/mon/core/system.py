#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""System utility helpers for terminal and reproducibility.

This module provides helpers for terminal control and for setting reproducible
seeds across Python, NumPy, and PyTorch to enable consistent terminal clearing
and experiment reproducibility in the codebase.
"""

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
# REPRODUCIBILITY & DETERMINISM
# ==============================================================================

# --- Global Seeding (Python, NumPy, and PyTorch RNG synchronization) ---
def set_random_seed(seed: int | tuple[int, int]):
    """Set random seeds for reproducibility.

    Use the provided seed or sample from a two-element range to set the Python,
    NumPy, and PyTorch RNGs and the PYTHONHASHSEED environment variable.

    Args:
        seed: Single integer seed or a two-element range (min, max) from which
            a seed will be sampled.
    """
    if isinstance(seed, Sequence):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================================================================
# SYSTEM & TERMINAL CONTROL
# ==============================================================================

# --- Shell Utilities (OS-agnostic terminal management) ---
def clear_terminal():
    """Clear the terminal screen.

    Issue the platform-specific command to clear the terminal display.
    """
    if platform.system() == "Windows":
        os.system("cls")
    elif platform.system() in ["Darwin", "Linux"]:
        os.system("clear")
