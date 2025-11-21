#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The ``mon`` framework.

The organization structure of this framework is inspired by the taxonomy
defined in https://www.ibm.com/think/machine-learning#605511093

2025.11.08: I am still wondering what is the best way to expose the sub-packages.
"""

__author__  = "Long H. Pham"
__version__ = "2.9.1"

import time
__start = time.time()

# Flat exposed APIs (core, high-frequency used functions)
from .core import *
from .training import build_dataloader, build_dataset

# Hierarchical exposed APIs (sub-packages)
from . import (
    nn,
    training as trn  # Alias for convenience
)
from .training import (
    albumentations,
    data,
    losses,
    metrics,
    optims,
)

__end = time.time()
console.log(f"`mon` loaded in: {__end - __start:.4f} seconds.")


# Keep specialized sub-packages lazy-loaded
def preload(verbose: bool = True):
    """Preload the specialized sub-packages of ``mon`` framework."""
    start = time.time()
    
    import mon.cv
    import mon.genai
    import mon.datasets
    
    end = time.time()
    if verbose:
        console.log(f"`mon-dev` loaded in: {end - start:.4f} seconds.")
