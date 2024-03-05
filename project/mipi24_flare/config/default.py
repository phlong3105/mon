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
    Task.LES,
]

# List all models that are used in this project.
MODELS = [

]
# If unsure, run the following script:
# mon.print_table(mon.MODELS | mon.MODELS_EXTRA)

# List all datasets that are used in this project.
DATASETS = [
    "mipi24_flare",
]
# If unsure, run the following script:
# mon.print_table(mon.DATASETS | mon.DATASETS_EXTRA)
