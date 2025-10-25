#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZID model prediction pipeline for image dehazing.

References:
    - Paper: "Zero-Shot Image Dehazing," IEEE TIP 2020.
    - Code: https://github.com/XLearning-SCU/2020-TIP-ZID
"""

__all__ = [
    "ZID",
]

from .model import ZID
