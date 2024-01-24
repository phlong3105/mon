#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions.
"""

from __future__ import annotations

__all__ = [
	"set_random_seed",
]

import random

import numpy as np
import torch


def set_random_seed(seed):
	"""Set random seeds."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
