#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module handles configuration files."""

from __future__ import annotations

__all__ = [
    "load_config",
]

from mon.foundation import file_handler, pathlib


# region Config

def load_config(cfg: dict | pathlib.Path | str) -> dict:
    """Load a configuration dictionary from the given :param:`cfg`. If it is a
    file, load its contents.
    """
    if isinstance(cfg, dict):
        data = cfg
    elif isinstance(cfg, pathlib.Path | str):
        data = file_handler.read_from_file(path=cfg)
    else:
        raise TypeError(
            f"cfg must be a dict or a path to the config file, but got {cfg}."
        )
    if data is None:
        raise IOError(f"No configuration is found at {cfg}.")
    return data

# endregion
