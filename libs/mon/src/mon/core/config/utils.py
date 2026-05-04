#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Config Utilities.

This module provides utilities for loading and managing configurations from YAML
files.
"""

from __future__ import annotations

__all__ = [
    "load_config",
]

import os

from box import Box

from mon.core.constants import K
from mon.core.path import Path
from mon.core.utils import merge_dicts


# ==============================================================================
# region INPUT
# ==============================================================================

def load_config(path: Path | None = None, config: dict | None = None) -> Box:
    """Load configurations from a YAML file.

    Args:
        path (Path | None, optional): Path to the YAML configuration file.
            Defaults to None.
        config (dict | None, optional): Additional configuration to merge with
            the loaded config. Defaults to empty dict.

    Returns:
        Box: The loaded and merged configuration as a Box for easy attribute access.
    """
    # 1. Normalize inputs
    config: dict = config or {}

    # 2. Load configurations from the YAML file
    if path:
        path: Path = Path(path).normalize()
        if not path.has_ext(".yaml", ".yml", exists=True):
            raise TypeError(
                f"Expected 'path' to be a valid configuration file path, "
                f"but got: {type(path).__name__}.",
            )
        file_config = Box.from_yaml(filename=path)
    else:
        file_config = {}

    # Load additional configurations from base YAML files if specified in
    # the ``__include__`` key
    if K.INCLUDE_KEY in file_config:
        base_yamls = list(file_config[K.INCLUDE_KEY])
    elif K.INCLUDE_KEY in config:
        base_yamls = list(file_config[K.BASE_KEY])
    else:
        base_yamls = []

    for base_yaml in base_yamls:
        if base_yaml.startswith("~"):
            base_yaml = os.path.expanduser(base_yaml)
        if not base_yaml.startswith("/"):
            base_yaml = path.parent / base_yaml

        base_config = Box.from_yaml(filename=str(base_yaml))
        merge_dicts(config, base_config)

    # Merge the provided config with the loaded config
    config = merge_dicts(config, file_config)

    # Return the merged configuration as a Box for easy attribute access
    return Box(config)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
