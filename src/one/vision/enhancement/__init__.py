#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import importlib
from pathlib import Path

current_dir = Path(__file__).resolve().parent
files       = list(current_dir.rglob("*.py"))
for f in files:
    module = f.stem
    if module == "__init__":
        continue
    importlib.import_module(f"one.vision.enhancement.{module}")
