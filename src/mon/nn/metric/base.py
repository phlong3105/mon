#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for all metrics, and the
corresponding helper functions.
"""

from __future__ import annotations

__all__ = [
    "BootStrapper",
    "CatMetric",
    "ClasswiseWrapper",
    "MaxMetric",
    "MeanMetric",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "RunningMean",
    "RunningSum",
    "SumMetric",
]

from abc import ABC
from typing import Literal

import torchmetrics


# region Base

class Metric(torchmetrics.Metric, ABC):
    """The base class for all loss functions.

    Args:
        mode: One of: ``'FR'`` or ``'NR'``. Default: ``'FR'``.
        lower_is_better: Default: ``False``.
    """
    
    def __init__(
        self,
        mode           : Literal["FR", "NR"] = "FR",
        lower_is_better: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._mode            = mode
        self._lower_is_better = lower_is_better

# endregion


# region Aggregation

MetricCollection = torchmetrics.MetricCollection

CatMetric        = torchmetrics.CatMetric
MaxMetric        = torchmetrics.MaxMetric
MeanMetric       = torchmetrics.MeanMetric
MinMetric        = torchmetrics.MinMetric
RunningMean      = torchmetrics.RunningMean
RunningSum       = torchmetrics.RunningSum
SumMetric        = torchmetrics.SumMetric

# endregion


# region Wrapper

BootStrapper         = torchmetrics.BootStrapper
ClasswiseWrapper     = torchmetrics.ClasswiseWrapper
MetricTracker        = torchmetrics.MetricTracker
MinMaxMetric         = torchmetrics.MinMaxMetric
MultioutputWrapper   = torchmetrics.MultioutputWrapper
MultitaskWrapper     = torchmetrics.MultitaskWrapper

# endregion
