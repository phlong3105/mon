#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for type manipulation and conversion,
including creating combinations, merging dictionaries, sorting collections,
and converting values to specific types or tuples of specified lengths.
"""

__all__ = [
    "create_combinations",
    "is_float",
    "is_int",
    "merge_dicts",
    "sort",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_ntuple",
    "to_str",
    "unique",
]

import copy
import itertools
import re
from typing import Any, Callable, Collection, Iterable, Sequence

import box


# ----- Create -----
def create_combinations(seq: Sequence) -> list:
    """Create all combinations of elements in a sequence.

    Args:
        seq: Input sequence (e.g., :math:`[1,2,3]`).

    Returns:
        A ``list`` of all combinations of elements
        (e.g., :math:`[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]`).
    """
    x = copy.deepcopy(seq)
    x = list(x)
    x = [list(comb) for r in range(1, len(x) + 1) for comb in itertools.combinations(x, r)]
    return x


# ----- Manipulation -----
def merge_dicts(*dicts: dict) -> dict | box.Box:
    """Merges multiple ``dict``, prioritizing later ``dict`` will overwrite
    earlier ones.
    
    Args:
        *dicts: Dictionaries to merge. The first dictionary is the base.
    """
    merged = dicts[0]
    for i in range(1, len(dicts)):
        # Filter out None, "None", and empty string values
        ns_d = {k: v for k, v in dicts[i].items() if v not in [None, "None", ""]}
        merged.update(ns_d)
    return box.Box(**merged)


def sort(col: Collection, reverse: bool = False) -> Any:
    """Sorts a collection (``list``, ``tuple``, or ``dict``) in ascending order by default."""
    if isinstance(col, list | tuple):
        return type(col)(sorted(col, reverse=reverse))
    if isinstance(col, dict):
        sorted_items = sorted(col.items(), key=lambda item: item[0], reverse=reverse)
        return dict(sorted_items)
    raise TypeError(f"``col`` must be an iterable or a dict, got {type(col)}.")


def unique(seq: Sequence) -> Sequence:
    """Returns unique items from a sequence, preserving order.

    Args:
        seq: Input sequence (``list`` or ``tuple``) that may contain duplicates.

    Returns:
        Unique sequence matching type of ``seq``.

    Raises:
        TypeError: If ``seq`` is not a ``list`` or ``tuple``.
    """
    if not isinstance(seq, list | tuple):
        raise TypeError(f"``seq`` must be a list or tuple, got {type(seq).__name__}.")
    return type(seq)(set(seq))


# ----- Convert -----
def to_int(int_or_str: Any) -> int | None:
    """Converts a value to an integer.

    Args:
        int_or_str: Value to convert.

    Returns:
        A converted ``int`` or ``None`` if ``value`` is ``None``.

    Raises:
        ValueError: If ``int_or_str`` cannot be converted to an integer.
    """
    if int_or_str is None:
        return None
    try:
        return int(int_or_str)
    except (ValueError, TypeError):
        raise ValueError(f"``int_or_str`` must be convertible to int, "
                         f"got {int_or_str} ({type(int_or_str).__name__}).")


def to_float(float_or_str: Any) -> float | None:
    """Converts a value to a float.

    Args:
        float_or_str: Value to convert.

    Returns:
        A converted ``float`` or ``None`` if ``value`` is ``None``.

    Raises:
        ValueError: If ``value`` cannot be converted to a float.
    """
    if float_or_str is None:
        return None
    try:
        return float(float_or_str)
    except (ValueError, TypeError):
        raise ValueError(f"``float_or_str`` must be convertible to float, "
                         f"got {float_or_str} ({type(float_or_str).__name__}).")


def to_str(value: Any, sep: str = ",") -> str:
    """Converts a value to a ``str``, joining iterable elements with a delimiter.

    Args:
        value: Value to convert.
        sep: Delimiter for separating elements. Default: ``","``.

    Returns:
        A ``str`` representation of ``value``, with elements joined by ``sep``
        if iterable.
    """
    if isinstance(value, dict):
        items = [str(item) for item in value.values()]
    elif isinstance(value, list | tuple):
        items = [str(item) for item in value]
    else:
        return str(value) if value else ""
    
    return sep.join(items) if items else ""


def to_list(value: Any, sep = (",", ";", ":")) -> list:
    """Converts a ``tuple``, ``dict``, or ``str`` to ``list``, splitting strings
    by delimiters if needed.

    Args:
        value: Value to convert.
        sep: Delimiters for splitting. Default: ``(",", ";", ":")``.

    Returns:
        A ``list`` representation of ``value``.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, str):
        stripped = re.sub(r"^\s+|\s+$|\s", "", value)
        for delimiter in sep:
            if delimiter in stripped:
                return stripped.split(delimiter)
        return [stripped]
    return [value] if value else []


def to_int_list(value: Any, sep = (",", ";", ":")) -> list[int]:
    """Converts a value to a ``list`` of integers, splitting strings by delimiters.

    Args:
        value: Value to convert.
        sep: Delimiters for splitting. Default: ``(",", ";", ":")``.

    Returns:
        A ``list`` of integers.
    """
    return [int(item) for item in to_list(value, sep=sep)]


def to_float_list(value: Any, sep = (",", ";", ":")) -> list[float]:
    """Converts a value to a list of floats, splitting strings by delimiters.

    Args:
        value: Value to convert.
        sep: Delimiters for splitting. Default: ``(",", ";", ":")``.

    Returns:
        A ``list`` of floats
    """
    return [float(item) for item in to_list(value, sep=sep)]


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Defines a function to convert an input to a tuple of length ``n``.

    Args:
        n: The tuple length.

    Returns:
        A function converting input to tuple of length ``n`` via replication or
        truncation.
    """
    def parse(x: Any) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, str | bytes):
            items = tuple(x)
            return tuple(items * (n // len(items) + 1))[:n] if len(items) == 1 else items[:n]
        return tuple(itertools.repeat(x, n))
    return parse


# ----- Validation -----
def is_int(int_or_str: Any) -> bool:
    """Checks if a value can be converted to an integer.

    Args:
        int_or_str: Value to check.

    Returns:
        ``True`` if convertible to ``int``, ``False`` otherwise.
    """
    try:
        int(int_or_str)
        return True
    except (ValueError, TypeError):
        return False


def is_float(float_or_str: Any) -> bool:
    """Checks if a value can be converted to a float.

    Args:
        float_or_str: Value to check.

    Returns:
        ``True`` if convertible to ``float``, ``False`` otherwise.
    """
    try:
        float(float_or_str)
        return True
    except (ValueError, TypeError):
        return False
