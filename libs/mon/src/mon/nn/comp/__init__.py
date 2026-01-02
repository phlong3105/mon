#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Neural network components.

This package contains various modular components which can be assembled together
to form concrete neural network architectures.

Notes:
    - Design Pattern: Multiple Template Methods.
    - Goal: Encapsulate multiple "Template Methods" for neural network components.
    - Structure:
        ::
            
            comp/
            ├── __init__.py        # Unified entry point
            ├── template_A/        # Break into a "Template Method" if the module becomes too large
            │   ├── __init__.py    # Registry and factory logic
            │   ├── base.py        # Base classes and mixins
            │   ├── basic.py       # Basic functionalities
            │   ├── ...
            │   ├── utils.py       # Utility functions and helpers
            │   └── external/      # Expose external libraries
            │       └── ...
            └── ...
"""

# __all__ = []  # Prevent accidental imports of submodules.

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
