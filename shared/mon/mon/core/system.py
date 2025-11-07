#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements system-related utilities such as clearing the terminal
and setting random seeds for reproducibility.
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


# ----- Seed -----
def set_random_seed(seed: int | tuple[int, int]) -> None:
    """Sets random seeds for various libraries.

    Args:
        seed: An ``int``, or a ``tuple`` of :math:`(min, max)` for random selection.
    """
    if isinstance(seed, Sequence):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----- Terminal -----
def clear_terminal():
    """Clears the terminal screen."""
    if platform.system() == "Windows":
        os.system("cls")
    elif platform.system() in ["Darwin", "Linux"]:
        os.system("clear")
