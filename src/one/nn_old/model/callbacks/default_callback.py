#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning import callbacks

from one.core import CALLBACKS

# MARK: - Register

CALLBACKS.register(name="backbone_finetuning",             module=callbacks.BackboneFinetuning)
CALLBACKS.register(name="device_stats_monitor",            module=callbacks.DeviceStatsMonitor)
CALLBACKS.register(name="early_stopping",                  module=callbacks.EarlyStopping)
CALLBACKS.register(name="gpu_stats_monitor",               module=callbacks.GPUStatsMonitor)
CALLBACKS.register(name="gradient_accumulation_scheduler", module=callbacks.GradientAccumulationScheduler)
CALLBACKS.register(name="learning_rate_monitor",           module=callbacks.LearningRateMonitor)
CALLBACKS.register(name="model_checkpoint",                module=callbacks.ModelCheckpoint)
CALLBACKS.register(name="model_pruning",                   module=callbacks.ModelPruning)
CALLBACKS.register(name="model_summary",                   module=callbacks.ModelSummary)
CALLBACKS.register(name="quantization_aware_training",     module=callbacks.QuantizationAwareTraining)
CALLBACKS.register(name="stochastic_weight_averaging",     module=callbacks.StochasticWeightAveraging)
