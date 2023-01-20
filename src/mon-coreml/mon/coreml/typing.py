#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines type aliases used throughout :mod:`mon.coreml` package.
"""

from __future__ import annotations

__all__ = [
    "CallbackType", "CallbacksType", "ClassLabelsType", "EpochOutput",
    "LRSchedulerType", "LRSchedulersType", "LoggerType", "LoggersType",
    "LossType", "LossesType", "MetricType", "MetricsType", "ModelPhaseType",
    "OptimizerType", "OptimizersType", "ReductionType", "StepOutput",
    "TransformType", "TransformsType",
    # Extend :mod:`mon.foundation.typing`
    "CallableType", "ConfigType", "DictType", "Float1T", "Float2T", "Float3T",
    "Float4T", "Float5T", "Float6T", "FloatAnyT", "Floats", "ImageFormatType",
    "Int1T", "Int2T", "Int3T", "Int4T", "Int5T", "Int6T", "IntAnyT", "Ints",
    "MemoryUnitType", "PathType", "PathsType", "Strs", "VideoFormatType",
]

from typing import Any, Collection, Sequence, TypeAlias

import torch

from mon.coreml import callback, constant, data, logger, loss, metric, optimizer
from mon.foundation.typing import *

StepOutput      : TypeAlias = torch.Tensor | dict[str, Any]
EpochOutput     : TypeAlias = list[StepOutput]
PretrainedType  : TypeAlias = DictType | PathType | bool

CallbackType    : TypeAlias = callback.Callback | DictType
CallbacksType   : TypeAlias = CallbackType | Collection[CallbackType]

ModelPhaseType  : TypeAlias = str | int | constant.ModelPhase
ReductionType   : TypeAlias = str | int | constant.Reduction

ClassLabelsType : TypeAlias = data.ClassLabels | list | PathType | DictType
TransformType   : TypeAlias = data.Transform | DictType
TransformsType  : TypeAlias = TransformType | Sequence[TransformType] | data.ComposeTransform

LoggerType      : TypeAlias = logger.Logger | DictType
LoggersType     : TypeAlias = LoggerType | Collection[LoggerType]

LossType        : TypeAlias = loss.Loss | DictType
LossesType      : TypeAlias = LossType | Collection[LossType]

MetricType      : TypeAlias = metric.Metric | DictType
MetricsType     : TypeAlias = MetricType | Collection[MetricType]

LRSchedulerType : TypeAlias = optimizer.LRScheduler | DictType
LRSchedulersType: TypeAlias = LRSchedulerType | Collection[LRSchedulerType]
OptimizerType   : TypeAlias = optimizer.Optimizer | DictType
OptimizersType  : TypeAlias = OptimizerType | Collection[OptimizerType]
