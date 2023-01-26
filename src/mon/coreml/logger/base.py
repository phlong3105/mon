#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all loggers, and the corresponding
helper functions.
"""

from __future__ import annotations

__all__ = [
    "CSVLogger", "CometLogger", "Logger", "MLFlowLogger", "NeptuneLogger",
    "WandbLogger",
]

from lightning.pytorch import loggers

from mon.coreml import constant

# region Logger

Logger        = loggers.Logger
CSVLogger     = loggers.CSVLogger
CometLogger   = loggers.CometLogger
MLFlowLogger  = loggers.MLFlowLogger
NeptuneLogger = loggers.NeptuneLogger
WandbLogger   = loggers.WandbLogger

constant.LOGGER.register(name="csv_logger",     module=CSVLogger)
constant.LOGGER.register(name="comet_logger",   module=CometLogger)
constant.LOGGER.register(name="mlflow_logger",  module=MLFlowLogger)
constant.LOGGER.register(name="neptune_logger", module=NeptuneLogger)
constant.LOGGER.register(name="wandb_logger",   module=WandbLogger)
constant.LOGGER.register(name="csv",            module=CSVLogger)
constant.LOGGER.register(name="comet",          module=CometLogger)
constant.LOGGER.register(name="mlflow",         module=MLFlowLogger)
constant.LOGGER.register(name="neptune",        module=NeptuneLogger)
constant.LOGGER.register(name="wandb",          module=WandbLogger)

# endregion
