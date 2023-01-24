#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python :mod:`math` module."""

from __future__ import annotations

from math import *


def make_divisible(x: int | float, divisor: int) -> int:
    """Make a number :param:`x` evenly divisible by a :param:`divisor`."""
    return ceil(x / divisor) * divisor
