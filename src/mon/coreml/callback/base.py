#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all callbacks, and the
corresponding helper functions.
"""

from __future__ import annotations

__all__ = [
    "BackboneFinetuning", "BaseFinetuning", "BasePredictionWriter",
    "BatchSizeFinder", "Callback", "Checkpoint", "DeviceStatsMonitor",
    "EarlyStopping", "GradientAccumulationScheduler", "LambdaCallback",
    "LearningRateFinder", "LearningRateMonitor", "ModelPruning", "ModelSummary",
    "QuantizationAwareTraining", "StochasticWeightAveraging", "Timer",
]

from lightning.pytorch import callbacks

from mon.coreml import constant

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
QuantizationAwareTraining     = callbacks.QuantizationAwareTraining
StochasticWeightAveraging     = callbacks.StochasticWeightAveraging
Timer                         = callbacks.Timer

constant.CALLBACK.register(name="backbone_finetuning",             module=BackboneFinetuning)
constant.CALLBACK.register(name="batch_size_finder",               module=BatchSizeFinder)
constant.CALLBACK.register(name="device_stats_monitor",            module=DeviceStatsMonitor)
constant.CALLBACK.register(name="early_stopping",                  module=EarlyStopping)
constant.CALLBACK.register(name="gradient_accumulation_scheduler", module=GradientAccumulationScheduler)
constant.CALLBACK.register(name="learning_rate_finder",            module=LearningRateFinder)
constant.CALLBACK.register(name="learning_rate_monitor",           module=LearningRateMonitor)
constant.CALLBACK.register(name="model_pruning",                   module=ModelPruning)
constant.CALLBACK.register(name="model_summary",                   module=ModelSummary)
constant.CALLBACK.register(name="quantization_aware_training",     module=QuantizationAwareTraining)
constant.CALLBACK.register(name="stochastic_weight_averaging",     module=StochasticWeightAveraging)
constant.CALLBACK.register(name="timer",                           module=Timer)
# CALLBACKS.register(name="tune_report_callback",            module=TuneReportCallback)

# endregion
