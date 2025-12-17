#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements InDi-Deband model for image/video debanding

References:
    - Paper:
    - Code: https://github.com/ksasso1028/indi-debanding
"""

__all__ = [
    "InDiDeband",
]

from .indi import get_indi_step, indi_transform, sample
from .model import InDiDeband
