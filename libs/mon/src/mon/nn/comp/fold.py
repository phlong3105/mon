#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for fold and unfold operations.

This module implements classes for folding and unfolding tensors in neural
networks.
"""

__all__ = [
    "Fold",
    "Unfold",
]

from torch.nn.modules.fold import *
