#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""``mon`` framework.

This package provides a unified library for research and development. It mainly
covers computer vision and artificial intelligence.
"""

__author__ = "Long H. Pham"
__version__ = "2.10.0"

import time

_start_time = time.time()

from .core import *
# from . import cv, dataset, genai, metrics, nlp, nn

# from .training import *
# from . import nn
# import mon.training

_end_time = time.time()
log(f"`mon` loaded in: {_end_time - _start_time:.4f} seconds.")

# Keep specialized sub-packages lazy-loaded
'''
def preload(verbose: bool = True):
    """Preload specialized mon subpackages.

    Import optional heavy subpackages (cv, genai, datasets) to reduce first-call
    latency; optionally, log the elapsed load time when verbose is True.
    """
    start = time.time()

    # import mon.cv
    # import mon.genai
    # import mon.datasets

    end = time.time()
    if verbose:
        console.log(f"`mon-dev` loaded in: {end - start:.4f} seconds.")
'''
