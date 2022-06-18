#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os

__all__ = [
    "data_dir",
    "pretrained_dir",
]


# MARK: - Directories

__current_file   = os.path.abspath(__file__)                          # "workspaces/one/src/aic/utils.py"
source_root_dir  = os.path.dirname(__current_file)                    # "workspaces/one/src/aic"
content_root_dir = os.path.dirname(os.path.dirname(source_root_dir))  # "workspaces/one"
pretrained_dir   = os.path.join(content_root_dir, "pretrained")       # "workspaces/one/pretrained"
data_dir         = os.getenv("DATA_DIR", None)                        # In case we have set value in os.environ
if data_dir is None:
    data_dir = "/data"  # Run from Docker container
if not os.path.isdir(data_dir):
    data_dir = os.path.join(content_root_dir, "data")  # Run from `one` package
if not os.path.isdir(data_dir):
    data_dir = ""
