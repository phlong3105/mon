#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Simple Online and Realtime Tracking with a Deep
Association Metric (Deep SORT) tracker.
"""

from __future__ import annotations

__all__ = [
    "DeepSORT",
]

import numpy as np

from mon.globals import TRACKERS
from mon.vision.tracking import base

np.random.seed(0)


# region Helper Function

# endregion


# region DeepSORT

@TRACKERS.register(name="deepsort")
class DeepSORT(base.Tracker):
    """DeepSORT.
    
    See more: :class:`mon.vision.model.track.base.Tracker`.
    """
    pass
    
# endregion
