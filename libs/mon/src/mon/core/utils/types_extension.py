#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Type extension and conversion utility collection.

This module provides helpers for creating combinations, merging dictionaries,
sorting collections, and converting or validating types for downstream use.
"""

from __future__ import annotations

__all__ = [
    "create_combinations",
    "is_float",
    "is_int",
    "merge_dicts",
    "sort",
    "to_dict",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_ntuple",
    "to_str",
    "unique",
]

import itertools
import re
from typing import Any, Callable, Collection, Iterable, Sequence

import box


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Pre-compiled regex for stripping whitespace.
_WHITESPACE_RE = re.compile(r"\s+")

# endregion


# ==============================================================================
# VALIDATION
# ==============================================================================

def is_int(value: Any) -> bool:
    """Return True if a value can be safely converted to an integer.

    Args:
        value: The value to test.

    Returns:
        True if the value can be converted to an int, False otherwise.
    """
    if isinstance(value, int):
        return True
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def is_float(value: Any) -> bool:
    """Return True if a value can be safely converted to a float.

    Args:
        value: The value to test.

    Returns:
        True if the value can be converted to a float, False otherwise.
    """
    if isinstance(value, float):
        return True
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---
def unique(seq: Sequence) -> Sequence:
    """Return unique items from a sequence while preserving order and type.

    Args:
        seq: A list or tuple.

    Returns:
        A sequence of the same type containing unique items in order.

    Raises:
        TypeError: If `seq` is not a list or tuple.
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(f"Expected 'seq' to be a list or tuple, but got {type(seq).__name__}.")
    # dict.fromkeys is a highly efficient way to get unique items while preserving order.
    return type(seq)(dict.fromkeys(seq))


def create_combinations(seq: Sequence) -> list[list]:
    """Generate all non-empty combinations of elements from a sequence.

    Args:
        seq: A sequence of elements.

    Returns:
        A list of lists, where each inner list is a unique, non-empty
        combination of elements from the input sequence.
    """
    # Use itertools.chain.from_iterable for a memory-efficient and readable way
    # to generate combinations of all lengths.
    combs = itertools.chain.from_iterable(
        itertools.combinations(seq, r) for r in range(1, len(seq) + 1)
    )
    return [list(c) for c in combs]


# --- Aggregation ---


# endregion


# ==============================================================================
# region MUTATION
# ==============================================================================

# --- Alternation ---


# --- Rearrangement ---
def sort(col: Collection, reverse: bool = False) -> Any:
    """Return a sorted collection, preserving the input type where possible.

    Args:
        col: The collection to sort (list, tuple, or dict).
        reverse: If True, sort in descending order.

    Returns:
        A sorted collection of the same type as the input.

    Raises:
        TypeError: If `col` is not a list, tuple, or dict.
    """
    if isinstance(col, (list, tuple)):
        return type(col)(sorted(col, reverse=reverse))
    if isinstance(col, dict):
        # Sort by keys and reconstruct the dictionary.
        return {k: col[k] for k in sorted(col, reverse=reverse)}
    raise TypeError(f"Expected 'col' to be a list, tuple, or dict, "
                    f"but got {type(col).__name__}.")


# --- Addition ---


# --- Removal ---


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---
def to_int(value: Any) -> int | None:
    """Convert a value to an integer, returning None if the value is None.

    Args:
        value: The value to convert.

    Returns:
        An integer or None.

    Raises:
        ValueError: If the conversion fails for a non-None value.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Expected 'value' to be convertible to an integer, but got '{value}'.")


def to_float(value: Any) -> float | None:
    """Convert a value to a float, returning None if the value is None.

    Args:
        value: The value to convert.

    Returns:
        A float or None.

    Raises:
        ValueError: If the conversion fails for a non-None value.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Expected 'value' to be convertible to a float, but got '{value}'.")


def to_str(value: Any, sep: str = ",") -> str:
    """Convert a value to a string, joining iterables with a separator.

    Args:
        value: The value to convert (can be a dict, list, tuple, or scalar).
        sep: The separator to use when joining iterable elements.

    Returns:
        A string representation of the value.
    """
    if not value:
        return ""
    if isinstance(value, dict):
        return sep.join(map(str, value.values()))
    if isinstance(value, (list, tuple)):
        return sep.join(map(str, value))
    return str(value)


def to_list(value: Any, sep: str | tuple[str, ...] = (",", ";", ":")) -> list:
    """Normalize a value to a list.

    This function handles lists, tuples, dictionaries (by taking values), and
    strings (by splitting).

    Args:
        value: The input value.
        sep: A delimiter or tuple of delimiters to use for splitting strings.

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
        # Create a regex pattern from the separators to handle multiple delimiters.
        pattern = "|".join(map(re.escape, sep)) if isinstance(sep, tuple) else sep
        return [item for item in re.split(pattern, value) if item]
    return [value] if value is not None else []


def to_int_list(value: Any, sep: str | tuple[str, ...] = (",", ";", ":")) -> list[int]:
    """Convert a value to a list of integers.

    Args:
        value: The input value to normalize and convert.
        sep: Delimiters for splitting if the input is a string.

    Returns:
        A list of integers.
    """
    return list(int(i) for i in to_list(value, sep=sep))


def to_float_list(value: Any, sep: str | tuple[str, ...] = (",", ";", ":")) -> list[float]:
    """Convert a value to a list of floats.

    Args:
        value: The input value to normalize and convert.
        sep: Delimiters for splitting if the input is a string.

    Returns:
        A list of floats.
    """
    return [float(i) for i in to_list(value, sep=sep)]


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Return a function that converts an input to a tuple of length `n`.

    If the input is an iterable, it will be repeated or truncated to match `n`.
    If it's a scalar, it will be repeated `n` times.

    Args:
        n: The desired tuple length.

    Returns:
        A function that performs the conversion.
    """
    def parse(x: Any) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            items = list(x)
            # Efficiently repeat or truncate to the desired length.
            return tuple(itertools.islice(itertools.cycle(items), n))
        # For scalars, repeat the value n times.
        return (x,) * n
    return parse


def to_dict(value: Any) -> dict:
    """Convert a value to a dictionary.

    This function handles dictionaries and objects with a `.to_dict()` method
    (like `box.Box`).

    Args:
        value: The input value to convert.

    Returns:
        A dictionary.

    Raises:
        TypeError: If the conversion is not possible.
    """
    if isinstance(value, dict):
        return value
    # Use hasattr for safe duck-typing.
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"Expected 'value' to be a dict or have a 'to_dict' method, "
                    f"but got {type(value).__name__}.")


# --- Encoding ---


# --- Standardization ---


# --- Structural ---
def merge_dicts(*dicts: dict) -> box.Box:
    """Merge multiple dictionaries, filtering out None-like values.

    Later dictionaries override keys from earlier ones, unless the new value is
    `None`, `"None"`, or an empty string.

    Args:
        *dicts: A sequence of dictionaries to merge.

    Returns:
        A `box.Box` instance containing the merged key-value pairs.
    """
    if not dicts:
        return box.Box()

    merged       = dicts[0].copy()
    invalid_vals = {None, "None", ""}

    for d in dicts[1:]:
        # Use a dictionary comprehension for a concise, single-pass update.
        merged.update({k: v for k, v in d.items() if v not in invalid_vals})

    return box.Box(merged)


# --- Statistical ---


# --- Geometric ---


# endregion
