#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for SGZ.
"""

from __future__ import annotations

__all__ = [
    "get_no_gt_target",
    "resize_target",
    "weights_init",
]

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


# ==============================================================================
# region UTILITIES
# ==============================================================================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_no_gt_target(input: Tensor) -> Tensor:
    sfmx_inputs = F.log_softmax(input, dim=1)
    target = torch.argmax(sfmx_inputs, dim=1)
    return target


def resize_target(target: Tensor, size: int) -> np.ndarray:
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_CUBIC)
    return new_target

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
