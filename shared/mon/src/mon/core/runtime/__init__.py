#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for runtime utilities and CLI.

This package provides various utilities for command-line interface (CLI) handling,
configuration loading, argument parsing, and runtime summaries.
"""

__all__ = [
    "CLI_OPTIONS",
    "DEFAULT_ARGS",
    "RunCLI",
    "list_archs",
    "list_config_files",
    "list_datasets",
    "list_models",
    "list_weights_files",
    "load_config",
    "load_project_defaults",
    "parse_cli_args",
    "parse_config_file",
    "parse_data_dir",
    "parse_default_args",
    "parse_model_dir",
    "parse_model_fullname",
    "parse_output_dir",
    "parse_predict_args",
    "parse_save_dir",
    "parse_train_args",
    "parse_weights_dir",
    "parse_weights_file",
    "parse_weights_from_config",
    "print_run_summary",
]

from .menu_rich import RunCLI
from .options import CLI_OPTIONS, DEFAULT_ARGS
from .parse import (
    parse_cli_args,
    parse_default_args,
    parse_predict_args,
    parse_train_args,
)
from .utils import (
    list_archs,
    list_config_files,
    list_datasets,
    list_models,
    list_weights_files,
    load_config,
    load_project_defaults,
    parse_config_file,
    parse_data_dir,
    parse_model_dir,
    parse_model_fullname,
    parse_output_dir,
    parse_save_dir,
    parse_weights_dir,
    parse_weights_file,
    parse_weights_from_config,
    print_run_summary,
)
