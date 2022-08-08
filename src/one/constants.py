#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A place to store all global constants.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from one.core import Factory
from one.core import OptimizerFactory
from one.core import SchedulerFactory
from one.core import VisionBackend


# H1: - Directory --------------------------------------------------------------

__current_file   = Path(__file__).absolute()         # "workspaces/one/src/one/constants.py"
SOURCE_ROOT_DIR  = __current_file.parents[0]         # "workspaces/one/src/one"
CONTENT_ROOT_DIR = __current_file.parents[2]         # "workspaces/one"
PRETRAINED_DIR   = CONTENT_ROOT_DIR / "pretrained"   # "workspaces/one/pretrained"
DATA_DIR         = os.getenv("DATA_DIR", None)       # In case we have set value in os.environ
if DATA_DIR is None:
    DATA_DIR = Path("/data")                        # Run from Docker container
else:
    DATA_DIR = Path(DATA_DIR)
if not DATA_DIR.is_dir():
    DATA_DIR = CONTENT_ROOT_DIR / "data"            # Run from `one` package
if not DATA_DIR.is_dir():
    DATA_DIR = ""
    

# H1: - Factory-----------------------------------------------------------------

# one.nn
BACKBONES       = Factory(name="backbones")
CALLBACKS       = Factory(name="callbacks")
LAYERS          = Factory(name="layers")
LOGGERS         = Factory(name="loggers")
LOSSES          = Factory(name="losses")
METRICS         = Factory(name="metrics")
MODELS          = Factory(name="models")
MODULE_WRAPPERS = Factory(name="module_wrappers")
OPTIMIZERS      = OptimizerFactory(name="optimizers")
SCHEDULERS      = SchedulerFactory(name="schedulers")
# Misc
AUGMENTS        = Factory(name="augments")
DATAMODULES     = Factory(name="datamodules")
DATASETS        = Factory(name="datasets")
DISTANCES       = Factory(name="distance_functions")
DISTANCE_FUNCS  = Factory(name="distance_functions")
FILE_HANDLERS   = Factory(name="file_handler")
LABEL_HANDLERS  = Factory(name="label_handlers")
MOTIONS         = Factory(name="motions")
TRANSFORMS      = Factory(name="transforms")


# H1: - Misc -------------------------------------------------------------------

DEFAULT_CROP_PCT = 0.875
IMG_MEAN         = [0.485, 0.456, 0.406]
IMG_STD          = [0.229, 0.224, 0.225]
PI               = torch.tensor(3.14159265358979323846)
VISION_BACKEND   = VisionBackend.PIL
