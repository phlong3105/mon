#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python :mod:`math` module."""

from __future__ import annotations

from math import *


def make_divisible(x: int, divisor: int) -> int:
    """Make a number :param:`x` evenly divisible by a :param:`divisor`."""
    return ceil(x / divisor) * divisor


def get_hw(size: int | list[int]) -> list[int]:
    """Casts a size object to the standard :math:`[H, W]`.

    Args:
        size: A size of an image, windows, or kernels, etc.

    Returns:
        A size in :math:`[H, W]` format.
    """
    if isinstance(size, list | tuple):
        if len(size) == 3:
            if size[0] >= size[3]:
                size = size[0:2]
            else:
                size = size[1:3]
        elif len(size) == 1:
            size = [size[0], size[0]]
    else:
        size = [size, size]
    return size
