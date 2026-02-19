#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UI Components.

This package contains components for user interaction, including CLI and GUI
elements.

File Structure:
::

    ui/
    ├── __init__.py
    ├── console.py      # Console utilities and logging
    ├── progress.py     # Custom progress bars and download bars
    └── prompt.py       # Custom prompts for interactive CLI
"""

from __future__ import annotations

from .console import *
from .progress import *
from .prompt import *
