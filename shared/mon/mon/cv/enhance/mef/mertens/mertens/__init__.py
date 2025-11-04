#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Mertens et. al Exposure Fusion method.

References:
    - Paper: "Exposure Fusion," PG 2007.
    - Code: https://github.com/Jamy-L/Pytorch-Exposure-Fusion
"""

__all__ = [
    "Mertens",
    "mertens",
    "mertens_cv2",
]

from .model import Mertens, mertens, mertens_cv2
