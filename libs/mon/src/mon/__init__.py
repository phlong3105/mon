#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""``mon`` framework package.

Provide the top-level package for the mon framework, expose core and
high-frequency APIs, and offer lazy loading for heavy optional subpackages.
"""

__author__  = "Long H. Pham"
__version__ = "2.10.0"

import time
__start = time.time()

from .core import *
from .training import *
from . import nn
import mon.training

__end = time.time()
console.log(f"`mon` loaded in: {__end - __start:.4f} seconds.")


# Keep specialized sub-packages lazy-loaded
def preload(verbose: bool = True):
    """Preload specialized mon subpackages.

    Import optional heavy subpackages (cv, genai, datasets) to reduce first-call
    latency; optionally log the elapsed load time when verbose is True.
    """
    start = time.time()

    import mon.cv
    # import mon.genai
    import mon.datasets

    end = time.time()
    if verbose:
        console.log(f"`mon-dev` loaded in: {end - start:.4f} seconds.")
