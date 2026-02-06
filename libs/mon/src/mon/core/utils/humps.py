#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""String case conversion and validation utilities.

This module provides recursive converters and validators for various string
cases.
"""

from __future__ import annotations

__all__ = [
    "camelize",
    "decamelize",
    "dekebabize",
    "depascalize",
    "is_camelcase",
    "is_kebabcase",
    "is_pascalcase",
    "is_snakecase",
    "kebabize",
    "pascalize",
    "snakecase",
]

import re
from collections.abc import Mapping
from typing import Any, Callable


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Pre-compiled regex patterns for validation and splitting.
_CAMEL_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SNAKE_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_WORD_RE = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+")


# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

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


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---

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
