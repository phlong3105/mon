#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Callback.

This module implements the base classes for all callbacks, and the corresponding
helper functions.
"""

from __future__ import annotations

__all__ = [
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "BatchSizeFinder",
    "Callback",
    "Checkpoint",
    "DeviceStatsMonitor",
    "EarlyStopping",
    "GradientAccumulationScheduler",
    "LambdaCallback",
    "LearningRateFinder",
    "LearningRateMonitor",
    "ModelPruning",
    "ModelSummary",
    "OnExceptionCheckpoint",
    "SpikeDetection",
    "StochasticWeightAveraging",
    "TQDMProgressBar",
    "TimerCallback",
    # "TuneReportCallback",
    # "TuneReportCheckpointCallback",
]

# import ray.tune.integration.pytorch_lightning as ray
from lightning.pytorch import callbacks

from mon.globals import CALLBACKS

# region Callback

Callback                      = callbacks.Callback
BackboneFinetuning            = callbacks.BackboneFinetuning
BaseFinetuning                = callbacks.BaseFinetuning
BasePredictionWriter          = callbacks.BasePredictionWriter
BatchSizeFinder               = callbacks.BatchSizeFinder
Checkpoint                    = callbacks.Checkpoint
DeviceStatsMonitor            = callbacks.DeviceStatsMonitor
EarlyStopping                 = callbacks.EarlyStopping
GradientAccumulationScheduler = callbacks.GradientAccumulationScheduler
LambdaCallback                = callbacks.LambdaCallback
LearningRateFinder            = callbacks.LearningRateFinder
LearningRateMonitor           = callbacks.LearningRateMonitor
ModelPruning                  = callbacks.ModelPruning
ModelSummary                  = callbacks.ModelSummary
OnExceptionCheckpoint         = callbacks.OnExceptionCheckpoint
ProgressBar                   = callbacks.ProgressBar
StochasticWeightAveraging     = callbacks.StochasticWeightAveraging
SpikeDetection                = callbacks.SpikeDetection
TimerCallback                 = callbacks.Timer
TQDMProgressBar               = callbacks.TQDMProgressBar
# TuneReportCallback            = ray.TuneReportCallback
# TuneReportCheckpointCallback  = ray.TuneReportCheckpointCallback

CALLBACKS.register(name="backbone_finetuning"            , module=BackboneFinetuning)
CALLBACKS.register(name="batch_size_finder"              , module=BatchSizeFinder)
CALLBACKS.register(name="device_stats_monitor"           , module=DeviceStatsMonitor)
CALLBACKS.register(name="early_stopping"                 , module=EarlyStopping)
CALLBACKS.register(name="gradient_accumulation_scheduler", module=GradientAccumulationScheduler)
CALLBACKS.register(name="learning_rate_finder"           , module=LearningRateFinder)
CALLBACKS.register(name="learning_rate_monitor"          , module=LearningRateMonitor)
CALLBACKS.register(name="model_pruning"                  , module=ModelPruning)
CALLBACKS.register(name="model_summary"                  , module=ModelSummary)
CALLBACKS.register(name="on_exception_checkpoint"        , module=OnExceptionCheckpoint)
CALLBACKS.register(name="stochastic_weight_averaging"    , module=StochasticWeightAveraging)
CALLBACKS.register(name="spike_detection"                , module=SpikeDetection)
CALLBACKS.register(name="timer"                          , module=TimerCallback)
CALLBACKS.register(name="tqdm_progress_bar"              , module=TQDMProgressBar)
# CALLBACKS.register(name="tune_report_callback"           , module=TuneReportCallback)
# CALLBACKS.register(name="tune_report_checkpoint_callback", module=TuneReportCheckpointCallback)

# endregion
