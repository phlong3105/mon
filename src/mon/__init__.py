#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The MON package.

The sub-packages abstraction hierarchy is as follows:

- :mod:`foundation` implements the most ATOMIC, self-contained modules. As the
  name suggests, it is the foundation to build the core components on top.
- :mod:`core` implements core components including custom data types, creation
  patterns, data storage and persistence, networking, etc.
- :mod:`coreimage`
- :mod:`coreml`
- :mod:`vision`
"""

from __future__ import annotations

# Order-specific import. DO NOT CHANGE!!!
import mon.foundation
#
import mon.coreml
import mon.createml
import mon.globals
import mon.vision
from mon.coreml import *
from mon.createml import *
from mon.foundation import *
from mon.globals import *
from mon.vision import *

__version__ = "1.2.3"
