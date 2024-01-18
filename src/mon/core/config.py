#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module handles configuration files."""

from __future__ import annotations

__all__ = [
    "get_config_module",
    "load_config",
]

from mon.core import file, pathlib, rich

console = rich.console


# region Config

def get_config_module(
    project : str,
    name    : str,
    data    : str,
    variant : str | None = None,
    config  : str | None = None,
) -> str:
    """Get config module from given components in
    ``mon.src.config + <project/name/<config/name_variant_data>``.
    """
    project  = project  or ""
    name     = name     or ""
    variant  = variant  or ""
    data     = data     or ""
    config   = config   or ""

    # Config file
    if config == "":
        if name != "" and variant != "" and name in variant:
            config = variant
        else:
            config = f"{name}"
        config = f"{config}_{data}"
    config.replace("-", "_")

    # Config module
    from mon.globals import CONFIG_DIR
    config_module = str(CONFIG_DIR.name)
    if project != "":
        project       = str(project).replace("/", ".")
        config_module = f"{config_module}.{project}"
    if name != "":
        if config_module.split(".")[-1] != name:
            config_module = f"{config_module}.{name}"
    config_module = f"{config_module}.{config}"
    return config_module


def load_config(config: dict | pathlib.Path | str | None) -> dict | None:
    """Load configuration as a :class:`dict`. If it is a file, load its
    contents.
    
    Args:
        config: Can be a configuration :class:`dict`, or a file.
    
    Returns:
        A :class:`dict` of configuration.
    """
    if config is None:
        console.log(f":param:`config` is ``None``.")
        return None
    
    data = None
    if isinstance(config, dict):
        data = config
    elif isinstance(config, pathlib.Path | str):
        data = file.read_from_file(path=config)
    if data is None:
        raise IOError(f"No configuration is found at {config}.")
    return data

# endregion
