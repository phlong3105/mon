#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all loggers, and the corresponding
helper functions.
"""

from __future__ import annotations

__all__ = [
    "CSVLogger",
    "CometLogger",
    "Logger",
    "MLFlowLogger",
    "NeptuneLogger",
    "WandbLogger",
]

from lightning.pytorch import loggers

from mon.globals import LOGGERS

# region Logger

Logger        = loggers.Logger
CSVLogger     = loggers.CSVLogger
CometLogger   = loggers.CometLogger
MLFlowLogger  = loggers.MLFlowLogger
NeptuneLogger = loggers.NeptuneLogger
WandbLogger   = loggers.WandbLogger

LOGGERS.register(name="csv_logger"    , module=CSVLogger)
LOGGERS.register(name="comet_logger"  , module=CometLogger)
LOGGERS.register(name="mlflow_logger" , module=MLFlowLogger)
LOGGERS.register(name="neptune_logger", module=NeptuneLogger)
LOGGERS.register(name="wandb_logger"  , module=WandbLogger)
LOGGERS.register(name="csv"           , module=CSVLogger)
LOGGERS.register(name="comet"         , module=CometLogger)
LOGGERS.register(name="mlflow"        , module=MLFlowLogger)
LOGGERS.register(name="neptune"       , module=NeptuneLogger)
LOGGERS.register(name="wandb"         , module=WandbLogger)

# endregion
