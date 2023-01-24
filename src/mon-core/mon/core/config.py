#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module handles configuration files."""

from __future__ import annotations

__all__ = [
    "load_config",
]

import munch

from mon.core import file_handler, pathlib
from mon.core.typing import ConfigType


# region Config

def load_config(cfg: ConfigType) -> munch.Munch:
    """Load a configuration dictionary from the given :param:`cfg`. If it is a
    file, load its contents. If it is a dictionary, convert it to a namespace
    object using :class:`munch.Munch`.
    """
    if isinstance(cfg, pathlib.Path | str):
        d = file_handler.load_from_file(path=cfg)
    elif isinstance(cfg, munch.Munch | dict):
        d = cfg
    else:
        raise TypeError(
            f":param:`cfg` must be a :class:`munch.Munch`, `dict`, or a path to "
            f"the config file. But got: {cfg}."
        )
    if d is None:
        raise IOError(f"No configuration is found at: {cfg}.")
    if not isinstance(d, munch.Munch):
        d = munch.Munch.fromDict(d)
    return d
    
# endregion
