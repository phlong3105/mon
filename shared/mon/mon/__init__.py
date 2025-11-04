#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The MON framework."""

__author__  = "Long H. Pham"
__version__ = "2.9.0"


# Import core packages
import time
__start = time.time()

from mon.core import *
from mon.constants import *
from mon.nn import *
from mon.training import *
from mon.datasets import *

__end = time.time()
console.log(f"`mon` loaded in: {__end - __start:.4f} seconds.")


# Import development packages
def dev(verbose: bool = True):
    start = time.time()
    
    import mon.cv  # Register vision models
    
    end = time.time()
    if verbose:
        console.log(f"`mon-dev` loaded in: {end - start:.4f} seconds.")
