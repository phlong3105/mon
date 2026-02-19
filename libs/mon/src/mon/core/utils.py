#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""String & Basic Type Utilities.

This package contains helpers for string case conversion, type coercion,
collection utilities, and dictionary operations.
"""

from __future__ import annotations

__all__ = [
    "camelize",
    "create_combinations",
    "decamelize",
    "dekebabize",
    "depascalize",
    "find_unique",
    "is_camelcase",
    "is_dict_of",
    "is_dictlist_of",
    "is_float",
    "is_int",
    "is_kebabcase",
    "is_list_of",
    "is_list_of",
    "is_pascalcase",
    "is_snakecase",
    "is_valid_str",
    "kebabize",
    "merge_dicts",
    "pascalize",
    "snakecase",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_ntuple",
    "to_str",
    "truncate_string",
]

import collections
import itertools
import re
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Literal, Sequence

from box import Box

# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Pre-compiled regex patterns for validation and splitting.
_CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SNAKE_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_WORD_RE = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")

# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

# --- Scalars ---

def is_int(value: Any) -> bool:
    """Check if the input value can be converted to an integer."""
    if isinstance(value, int):
        return True
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def is_float(value: Any) -> bool:
    """Check if the input value can be converted to a float."""
    if isinstance(value, float):
        return True
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# --- Strings ---

def is_valid_str(value: Any) -> bool:
    """Check if the input value is a valid string."""
    try:
        value = str(value)
        value = _WHITESPACE_RE.sub("", str(value))
        return value.lower() not in ["", "none", "null", "nan", "inf"]
    except (ValueError, TypeError):
        return False


def is_camelcase(value: Any) -> bool:
    """Check if the input string is in camelCase."""
    return isinstance(value, str) and bool(_CAMEL_RE.match(value))


def is_pascalcase(value: Any) -> bool:
    """Check if the input string is in PascalCase."""
    return isinstance(value, str) and bool(_PASCAL_RE.match(value))


def is_kebabcase(value: Any) -> bool:
    """Check if the input string is in kebab-case."""
    return isinstance(value, str) and bool(_KEBAB_RE.match(value))


def is_snakecase(value: Any) -> bool:
    """Check if the input string is in snake_case."""
    return isinstance(value, str) and bool(_SNAKE_RE.match(value))


# --- Collections ---

def is_list_of(value: Any, type_: type) -> bool:
    """Check if the input value is a list of a specific type."""
    return (
        isinstance(value, list) and
        (
            # All elements are of the correct type
            all(isinstance(v, type_) for v in value)
            # Empty list is valid
            or not value
        )
    )


def is_listdict_of(value: Any, type_: type) -> bool:
    """Check if the input value is a list of dictionaries with values of a specific type."""
    if not isinstance(value, list):
        return False
    return all(is_dict_of(v, type_) for v in value)


def is_dict_of(value: Any, type_: type) -> bool:
    """Check if the input value is a dictionary with values of a specific type."""
    return (
        isinstance(value, dict) and
        (
            # All values are of the correct type
            all(isinstance(v, type_) for v in value.values())
            # Empty dict is valid
            or not value
        )
    )


def is_dictlist_of(value: Any, type_: type) -> bool:
    """Check if the input value is a dictionary with list values of a specific type."""
    if not isinstance(value, dict):
        return False
    return all(is_list_of(v, type_) for v in value.values())

# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---

def find_unique(seq: Sequence) -> Sequence:
    """Return unique items from ``seq`` while preserving order and type.

    Use ``dict.fromkeys`` for a highly efficient way to get unique items while
    preserving order.

    Raises:
        TypeError: If ``seq`` is not a list or tuple.
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            f"Expected 'seq' to be a list or tuple, but got {type(seq).__name__}.",
        )
    return type(seq)(dict.fromkeys(seq))


def create_combinations(seq: Sequence) -> list[list]:
    """Generate all non-empty combinations of elements from ``seq``.

    Use ``itertools.chain.from_iterable()`` for a memory-efficient and readable
    way to generate combinations of all lengths.
    """
    combs = itertools.chain.from_iterable(
        itertools.combinations(seq, r) for r in range(1, len(seq) + 1)
    )
    return [list(c) for c in combs]


# --- Aggregation ---


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---

def to_int(value: Any) -> int | None:
    """Convert ``value`` to an integer.

    Raises:
        ValueError: If the conversion fails for a non-None value.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Expected 'value' to be convertible to an integer, "
            f"but got 'f{value}'.",
        )


def to_float(value: Any) -> float | None:
    """Convert ``value`` to a float.

    Raises:
        ValueError: If the conversion fails for a non-None value.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(
            f"Expected 'value' to be convertible to a float, "
            f"but got '{value}'.",
        )


def to_str(value: Any, sep: str = ",") -> str:
    """Convert ``value`` to a string.

    Args:
        value (Any): Input value to convert.
        sep (str): Delimiter to use when joining collections. Defaults to ",".
    """
    if not value:
        return ""
    if isinstance(value, dict):
        return sep.join(map(str, value.values()))
    if isinstance(value, (list, tuple)):
        return sep.join(map(str, value))
    return str(value)


def to_list(value: Any, sep: str = ",|;|:") -> list:
    """Normalize ``value`` to a list.

    Args:
        value (Any): Input value to convert.
        sep (str): Delimiters for splitting if the input is a string.
            Defaults to ",|;|:"

    Returns:
        list: Normalized list.
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


def to_int_list(value: Any, sep: str = ",|;|:") -> list[int]:
    """Convert ``value`` to a list of integers.

    Args:
        value (Any): Input value to normalize and convert.
        sep (str): Delimiters for splitting if the input is a string.
            Defaults to ",|;|:"

    Returns:
        list[int]: List of integers.
    """
    return list(int(i) for i in to_list(value, sep=sep))


def to_float_list(value: Any, sep: str = ",|;|:") -> list[float]:
    """Convert ``value`` to a list of floats.

    Args:
        value (Any): Input value to normalize and convert.
        sep (str): Delimiters for splitting if the input is a string.
            Defaults to ",|;|:"

    Returns:
        list[float]: List of floats.
    """
    return [float(i) for i in to_list(value, sep=sep)]


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Return a function that converts an input to a tuple of length ``n``.

    Args:
        n (int): Desired tuple length.
    """

    def parse(x: Any) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            items = list(x)
            # Efficiently repeat or truncate to the desired length.
            return tuple(itertools.islice(itertools.cycle(items), n))
        # For scalars, repeat the value n times.
        return (x,) * n

    return parse


def pascalize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to PascalCase."""
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, pascalize)

    s = str(value)
    words = _separate_words(s)
    return "".join(word.capitalize() for word in words)


def camelize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to camelCase."""
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, camelize)

    s = str(value)
    words = _separate_words(s)
    pascalized = "".join(word.capitalize() for word in words)
    return pascalized[0].lower() + pascalized[1:] if pascalized else ""


def kebabize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to kebab-case."""
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, kebabize)

    s = str(value)
    words = _separate_words(s)
    return "-".join(word.lower() for word in words)


def snakecase(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to snake_case."""
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, snakecase)

    s = str(value)
    words = _separate_words(s)
    return "_".join(word.lower() for word in words)


# --- Structural ---

def merge_dicts(*dicts) -> dict:
    """Merge multiple dictionaries. Use the first dictionary as the base
    dictionary and update it with the remaining dictionaries, filtering out
    None-like values.

    Args:
        *dicts: Sequence of dictionaries to merge.

    Returns:
        dict: Merged dictionary.
    """
    # If no dictionaries are provided, return an empty dictionary.
    if not dicts:
        return {}

    merged = dicts[0].copy()
    invalid_vals = [None, "None", "", [], ()]

    # Iterate through all dictionaries provided after the first one
    for update_dict in dicts[1:]:
        for key, value in update_dict.items():
            if isinstance(value, collections.abc.Mapping):
                # The recursive call uses the existing nested dict (or a new empty one) as the base
                merged[key] = merge_dicts(merged.get(key, {}), value)
            else:
                if value in invalid_vals:
                    continue
                merged[key] = value

    return merged


def truncate_string(
    value: str,
    max_length: int = 80,
    side: Literal["left", "middle", "right"] = "middle"
) -> str:
    """Shortens a string for display purposes with an ellipsis in the middle.

    Args:
        value (str): The string to be truncated.
        max_length (int, optional): The maximum allowed length of the string
            including the ellipsis. Defaults to 80.
        side (Literal["left", "middle", "right"], optional): Where to place the
            ellipsis if truncation is needed. Defaults to "middle".
    """
    # Normalize inputs
    value = str(value)

    # Validate inputs
    if len(value) <= max_length:
        return value

    # Calculate how much to keep on each side of the "..."
    keep_len = (max_length - 3) // 2

    if side == "left":
        return "..." + value[-keep_len:]
    elif side == "right":
        return value[:keep_len] + "..."
    else:  # middle
        return value[:keep_len] + "..." + value[-keep_len:]


# --- Aliases for backward compatibility ---

decamelize = snakecase
depascalize = snakecase
dekebabize = snakecase

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _process_keys(data: Any, func: Callable) -> Any:
    """Apply a function to dictionary keys or list elements recursively."""
    if isinstance(data, Mapping):
        return {func(k): _process_keys(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [_process_keys(i, func) for i in data]
    return data


def _separate_words(string: str) -> list[str]:
    """Split a string into a list of words based on a case and separators."""
    return _WORD_RE.findall(string)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
