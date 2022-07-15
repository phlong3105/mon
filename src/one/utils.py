#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import os
import sys
from shutil import copyfile
from typing import Union

from munch import Munch

from one.core import create_dirs
from one.core import load_file

# MARK: - Directories

__current_file   = os.path.abspath(__file__)                          # "workspaces/one/src/one/utils.py"
source_root_dir  = os.path.dirname(__current_file)                    # "workspaces/one/src/one"
content_root_dir = os.path.dirname(os.path.dirname(source_root_dir))  # "workspaces/one"
pretrained_dir   = os.path.join(content_root_dir, "pretrained")       # "workspaces/one/pretrained"
data_dir         = os.getenv("DATA_DIR", None)                        # In case we have set value in os.environ
if data_dir is None:
    data_dir = "/data"  # Run from Docker container
if not os.path.isdir(data_dir):
    data_dir = os.path.join(content_root_dir, "data")  # Run from `one` package
if not os.path.isdir(data_dir):
    data_dir = ""


# MARK: - Process Config

def load_config(config: Union[str, dict, Munch]) -> Munch:
    """Load config as namespace.

	Args:
		config (str, dict, Munch):
			Config filepath that contains configuration values or the
			config dict.
	"""
    # NOTE: Load dictionary from file and convert to namespace using Munch
    if not isinstance(config, (dict, Munch, str)):
        raise TypeError(
            f"`config` must be a `dict` or a path to config file. "
            f"But got: {config}."
        )
    if isinstance(config, str):
        config_dict = load_file(path=config)
    else:
        config_dict = config
  
    if config_dict is None:
        raise IOError(f"No configuration is found at: {config}.")
   
    config = Munch.fromDict(config_dict)
    return config


def copy_config_file(config_file: str, dst: str):
    """Copy `config_file` to `dst` dir."""
    create_dirs(paths=[dst])
    copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
