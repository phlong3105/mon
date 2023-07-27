#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module handles configuration files."""

from __future__ import annotations

__all__ = [
    "load_config",
]

from mon.core import file, pathlib


# region Config

def load_config(config: dict | pathlib.Path | str) -> dict:
    """Load a configuration dictionary from the given :param:`cfg`. If it is a
    file, load its contents.
    """
    if isinstance(config, dict):
        data = config
    elif isinstance(config, pathlib.Path | str):
        data = file.read_from_file(path=config)
    else:
        raise TypeError(
            f"config must be a dict or a path to the config file, but got "
            f"{config}."
        )
    if data is None:
        raise IOError(f"No configuration is found at {config}.")
    return data
    
# endregion
