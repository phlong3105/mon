#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module offers configuration files handling capabilities."""

from __future__ import annotations

__all__ = [
    "load_config",
    "parse_config_file",
]

import importlib.util
from typing import Any

from mon.core import file, pathlib, rich

console       = rich.console
error_console = rich.error_console


# region Handling Config Files

def parse_config_file(
    config      : str | pathlib.Path,
    project_root: str | pathlib.Path
) -> pathlib.Path | None:
    # assert config not in [None, "None", ""]
    if config in [None, "None", ""]:
        error_console.log(f"No configuration given.")
        return None
    #
    config = pathlib.Path(config).config_file()
    if config.is_config_file():
        return config
    #
    config_dirs = [
        pathlib.Path(project_root),
        pathlib.Path(project_root) / "config"
    ]
    for config_dir in config_dirs:
        config_ = (config_dir / config.name).config_file()
        if config_.is_config_file():
            return config_
    #
    error_console.log(f"No configuration is found at {config}.")
    return None  # config

# endregion


# region Load

def load_config(config: Any) -> dict:
    if config is None:
        data = None
    elif isinstance(config, dict):
        data = config
    elif isinstance(config, pathlib.Path | str):
        config = pathlib.Path(config)
        if config.is_py_file():
            spec   = importlib.util.spec_from_file_location(str(config.stem), str(config))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            data  = {key: value for key, value in module.__dict__.items() if not key.startswith('__')}
        else:
            data = file.read_from_file(path=config)
    else:
        data = None
    
    if data is None:
        error_console.log(f"No configuration is found at {config}. Setting an empty dictionary.")
        data = {}
    return data

# endregion
