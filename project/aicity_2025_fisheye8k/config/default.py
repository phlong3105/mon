#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all common configuration settings used in this project."""

from __future__ import annotations

__all__ = [
    "DATASETS",
    "MODELS",
    "TASKS",
]

from mon.globals import Task

# List all tasks that are performed in this project.
TASKS = [
    Task.DETECTION,
]

# List all models that are used in this project.
MODELS = [
    "yolor_d6",
    "yolov7_e6e",
    "yolov8x",
    "yolov9_e",
]
# If unsure, run the following script:
# mon.print_table(mon.MODELS | mon.EXTRA_MODELS)

# List all datasets that are used in this project.
DATASETS = [
    "aicity_2024_fisheye8k",
]
# If unsure, run the following script:
# mon.print_table(mon.DATASETS | mon.DATASETS_EXTRA)
