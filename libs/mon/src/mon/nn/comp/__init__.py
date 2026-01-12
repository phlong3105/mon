#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Components for neural networks.

This package contains various modular components which can be assembled to form
concrete neural network implementations.

Notes:
    - Design Pattern: Multiple Template Methods.
    - Goal: Encapsulate multiple "Template Methods" for neural network components.
    - Structure:
        ::
            
            comp/
            ├── __init__.py        # Unified entry point
            ├── template/
            │   ├── __init__.py   # Registry and factory
            │   ├── api.py        # External APIs
            │   ├── base.py       # Base classes and mixins
            │   ├── basic.py      # Basic functionalities
            │   ├── ...
            │   ├── utils.py      # Utilities and helpers
            │   └── external/     # Integrate external libraries
            └── ...
"""

from __future__ import annotations

from .act import *
from .attention import *
from .conv import *
from .dropout import *
from .flatten import *
from .fold import *
from .fusion import *
from .linear import *
from .norm import *
from .padding import *
from .pooling import *
from .pos_enc import *
from .rnn import *
from .shuffle import *
from .sparse import *
from .transformer import *
from .upsampling import *
