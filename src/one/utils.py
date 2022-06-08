#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfile
from typing import Union

from munch import Munch

from one.io import create_dirs
from one.io import load_file

__all__ = [
    "content_root_dir",
    "copy_config_file",
    "data_dir",
    "load_config",
    "pretrained_dir",
    "source_root_dir",
]


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
    if isinstance(config, str):
        config_dict = load_file(path=config)
    elif isinstance(config, (dict, Munch)):
        config_dict = config
    else:
        raise ValueError(f"`config` must be a `dict` or a path to config file. "
                         f"But got: {config}.")
    if config_dict is None:
        raise ValueError(f"No configuration is found at: {config}.")
   
    config = Munch.fromDict(config_dict)
    return config


def copy_config_file(config_file: str, dst: str):
    """Copy `config_file` to `dst` dir."""
    create_dirs(paths=[dst])
    copyfile(config_file, os.path.join(dst, os.path.basename(config_file)))
