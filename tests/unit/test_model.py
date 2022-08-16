#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Loading Models
"""

from __future__ import annotations

from one.vision.enhancement import *


def test_models():
    """
    Test model creation.
    """
    zerodce = ZeroDCE(verbose=True)
