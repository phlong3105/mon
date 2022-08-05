#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from pytorch_lightning import loggers

from one.core import LOGGERS

# MARK: - Builder

# LOGGERS.register(name="tensorboard",        module=loggers.TensorBoardLogger)
# LOGGERS.register(name="tensorboard_logger", module=loggers.TensorBoardLogger)
# LOGGERS.register(name="TensorBoardLogger",  module=loggers.TensorBoardLogger)
LOGGERS.register(name="csv",              module=loggers.CSVLogger)
LOGGERS.register(name="csv_logger",       module=loggers.CSVLogger)
LOGGERS.register(name="CSVLogger",        module=loggers.CSVLogger)
LOGGERS.register(name="comet",            module=loggers.CometLogger)
LOGGERS.register(name="comet_logger",     module=loggers.CometLogger)
LOGGERS.register(name="CometLogger",      module=loggers.CometLogger)
LOGGERS.register(name="wandb",            module=loggers.WandbLogger)
LOGGERS.register(name="wandb_logger",     module=loggers.WandbLogger)
LOGGERS.register(name="WandbLogger",      module=loggers.WandbLogger)
LOGGERS.register(name="mlflow",           module=loggers.MLFlowLogger)
LOGGERS.register(name="mlflow_logger",    module=loggers.MLFlowLogger)
LOGGERS.register(name="MLFlowLogger",     module=loggers.MLFlowLogger)
LOGGERS.register(name="neptune",          module=loggers.NeptuneLogger)
LOGGERS.register(name="neptune_logger",   module=loggers.NeptuneLogger)
LOGGERS.register(name="NeptuneLogger",    module=loggers.NeptuneLogger)
LOGGERS.register(name="test_tube",        module=loggers.TestTubeLogger)
LOGGERS.register(name="test_tube_logger", module=loggers.TestTubeLogger)
LOGGERS.register(name="TestTubeLogger",   module=loggers.TestTubeLogger)
