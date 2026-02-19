#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Base Components.

This package contains base components for building datasets.

File Structure:
::

    base/
    ├── __init__.py
    ├── dataloader.py   # Dataloader
    ├── dataset.py      # Base datasets
    ├── image.py        # Base image datasets
    ├── mixins.py       # Dataset mixins
    ├── modality.py     # Base and concrete modalities (e.g., RGB, depth, etc.)
    └── video.py        # Base video datasets
"""

from __future__ import annotations

from .dataloader import *
from .dataset import *
from .image import *
from .mixins import *
from .modality import *
from .video import *
