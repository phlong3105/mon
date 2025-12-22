#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type extension and conversion utility collection.

This module provides helpers for creating combinations, merging dictionaries,
sorting collections, and converting or validating types for downstream use.
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


# ==============================================================================
# VALIDATION & SANITIZATION (Integrity Checks)
# ==============================================================================

# --- Scalar Checks ---
def is_int(int_or_str: Any) -> bool:
    """Return True if a value can be converted to an integer.

    Args:
        int_or_str: Value to test.

    Returns:
        True if conversion to int succeeds, False otherwise.
    """
    try:
        int(int_or_str)
        return True
    except (ValueError, TypeError):
        return False


def is_float(float_or_str: Any) -> bool:
    """Return True if a value can be converted to a float.

    Args:
        float_or_str: Value to test.

    Returns:
        True if conversion to float succeeds, False otherwise.
    """
    try:
        float(float_or_str)
        return True
    except (ValueError, TypeError):
        return False


# ==============================================================================
# TYPE CASTING & NORMALIZATION
# ==============================================================================

# --- Scalar Converters ---
def to_int(int_or_str: Any) -> int | None:
    """Convert a value to an integer or return None.

    Args:
        int_or_str: Value to convert.

    Returns:
        Converted integer or None.

    Raises:
        ValueError: If conversion fails.
    """
    if int_or_str is None:
        return None
    try:
        return int(int_or_str)
    except (ValueError, TypeError):
        raise ValueError(f"``int_or_str`` must be convertible to int, "
                         f"got {int_or_str} ({type(int_or_str).__name__}).")


def to_float(float_or_str: Any) -> float | None:
    """Convert a value to a float or return None.

    Args:
        float_or_str: Value to convert.

    Returns:
        Converted float or None.

    Raises:
        ValueError: If conversion fails.
    """
    if float_or_str is None:
        return None
    try:
        return float(float_or_str)
    except (ValueError, TypeError):
        raise ValueError(f"``float_or_str`` must be convertible to float, "
                         f"got {float_or_str} ({type(float_or_str).__name__}).")


def to_str(value: Any, sep: str = ",") -> str:
    """Convert a value to a string, joining iterables with a separator.

    Args:
        value: Value to stringify (dict, list, tuple, or scalar).
        sep: Separator used to join iterable elements.

    Returns:
        Joined string or empty string for falsy scalars.
    """
    if isinstance(value, dict):
        items = [str(item) for item in value.values()]
    elif isinstance(value, list | tuple):
        items = [str(item) for item in value]
    else:
        return str(value) if value else ""
    
    return sep.join(items) if items else ""


def to_list(value: Any, sep = (",", ";", ":")) -> list:
    """Normalize a value to a list.

    Args:
        value: Input value (list, tuple, dict, str, or scalar).
        sep: Delimiters to apply when splitting strings.

    Returns:
        A list representation of the input.
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


# --- Collection Converters ---
def to_int_list(value: Any, sep = (",", ";", ":")) -> list[int]:
    """Convert a value to a list of integers.

    Args:
        value: Input value to normalize and convert.
        sep: String delimiters for splitting when input is a str.

    Returns:
        Converted integers as a list.
    """
    return [int(item) for item in to_list(value, sep=sep)]


def to_float_list(value: Any, sep = (",", ";", ":")) -> list[float]:
    """Convert a value to a list of floats.

    Args:
        value: Input value to normalize and convert.
        sep: String delimiters for splitting when input is a str.

    Returns:
        Converted floats as a list.
    """
    return [float(item) for item in to_list(value, sep=sep)]


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Return a converter that produces an n-length tuple from input.

    Args:
        n: Desired tuple length.

    Returns:
        Function that converts input to a tuple of length n.
    """
    def parse(x: Any) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, str | bytes):
            items = tuple(x)
            return tuple(items * (n // len(items) + 1))[:n] if len(items) == 1 else items[:n]
        return tuple(itertools.repeat(x, n))
    return parse


# ==============================================================================
# TYPE CASTING & NORMALIZATION
# ==============================================================================

# --- Dictionary Merging ---
def merge_dicts(*dicts: dict) -> box.Box:
    """Merge multiple dictionaries while filtering None-like values.

    Later dictionaries override keys from earlier ones unless the later value
    is None, "None", or an empty string.

    Args:
        *dicts: Dictionaries to merge; first dict is treated as base.

    Returns:
        Merged mapping as a Box instance.
    """
    merged = dicts[0]
    for i in range(1, len(dicts)):
        # Filter out None, "None", and empty string values
        ns_d = {k: v for k, v in dicts[i].items() if v not in [None, "None", ""]}
        merged.update(ns_d)
    return box.Box(**merged)


# --- Set & Sequence Ops ---
def create_combinations(seq: Sequence) -> list:
    """Generate all non-empty combinations of elements.

    Args:
        seq: Sequence of elements.

    Returns:
        List of lists containing every non-empty combination (length 1..len(seq)).
    """
    x = copy.deepcopy(seq)
    x = list(x)
    x = [list(comb) for r in range(1, len(x) + 1) for comb in itertools.combinations(x, r)]
    return x


def sort(col: Collection, reverse: bool = False) -> Any:
    """Return a sorted collection preserving input type when possible.

    Args:
        col: Collection to sort (list, tuple, or dict).
        reverse: If True, sort in descending order.

    Returns:
        Sorted collection of the same type when supported.

    Raises:
        TypeError: If ``col`` has an unsupported type.
    """
    if isinstance(col, list | tuple):
        return type(col)(sorted(col, reverse=reverse))
    if isinstance(col, dict):
        sorted_items = sorted(col.items(), key=lambda item: item[0], reverse=reverse)
        return dict(sorted_items)
    raise TypeError(f"``col`` must be an iterable or a dict, got {type(col)}.")


def unique(seq: Sequence) -> Sequence:
    """Return unique items from a sequence preserving order and type.

    Args:
        seq: A list or tuple.

    Returns:
        A sequence of the same type containing unique items in insertion order.

    Raises:
        TypeError: If ``seq`` is not a list or tuple.
    """
    if not isinstance(seq, list | tuple):
        raise TypeError(f"``seq`` must be a list or tuple, got {type(seq).__name__}.")
    return type(seq)(set(seq))
