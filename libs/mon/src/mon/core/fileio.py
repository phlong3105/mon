#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic File I/O utilities.

This module provides low-level functions for reading and writing common text
formats like .json, .yaml, .txt, and .py
"""

from __future__ import annotations

__all__ = [
    "load_json",
    "load_txt",
    "load_yaml",
    "save_json",
    "save_txt",
    "save_yaml",
]

import json
from typing import Any

import yaml

from mon.core.pathlib import Path
from mon.core.utils import PathLike


# ==============================================================================
# region INPUT
# ==============================================================================

def load_json(path: PathLike) -> dict | list:
    """Load data from a JSON file."""
    path = Path(path).normalize(exist=True)

    if not path.is_json_file():
        raise ValueError(f"Expected a valid JSON file, but got '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: PathLike) -> dict | list:
    """Load data from a YAML file."""
    path = Path(path).normalize(exist=True)

    if not path.is_yaml_file():
        raise ValueError(f"Expected a valid YAML file, but got '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_txt(path: PathLike) -> list[str]:
    """Read lines from a text file."""
    path = Path(path).normalize(exist=True)

    if not path.is_txt_file():
        raise ValueError(f"Expected a valid text file, but got '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


# endregion


# ==============================================================================
# region OUTPUT
# ==============================================================================

def save_json(
    data: Any,
    path: PathLike,
    indent: int = 4,
    overwrite: bool = True,
):
    """Save data to a JSON file."""
    path = Path(path).normalize(mkdir=True)
    if not overwrite and path.exists():
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def save_yaml(
    data: Any,
    path: PathLike,
    overwrite: bool = True,
):
    """Save data to a YAML file."""
    path = Path(path).normalize(mkdir=True)
    if not overwrite and path.exists():
        return
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_txt(
    data: list[str] | str,
    path: PathLike,
    overwrite: bool = True,
):
    """Save strings to a text file."""
    path = Path(path).normalize(mkdir=True)
    if not overwrite and path.exists():
        return
    if isinstance(data, str):
        data = [data]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(data))


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
