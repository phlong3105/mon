#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Architectures.

This package includes base meta-architectures used across domains.

File Structure:
::

    models/
    ├── __init__.py
    ├── backbone/       # Feature extractors
    ├── head/           # Output heads
    ├── neck/           # Feature aggregators
    ├── repr/           # Representation learning
    └── mixins/         # Model mixins
"""

from __future__ import annotations

from .backbone import *
from .head import *
from .mixins import *
from .neck import *
from .repr import *
from .upsample import *
