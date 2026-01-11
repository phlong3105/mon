#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""String case conversion and validation utilities.

This module provides recursive converters and validators for camelCase,
PascalCase, kebab-case, and snake_case, supporting structured data such as
dictionaries and lists.
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
_CAMEL_RE  = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_KEBAB_RE  = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_SNAKE_RE  = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_WORD_RE   = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+")

# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================

def is_camelcase(value: Any) -> bool:
    """Return True if the input string is in camelCase.

    Args:
        value: The input to check.

    Returns:
        True if ``value`` is a string in camelCase, False otherwise.

    Examples:
        >>> is_camelcase("camelCase")
        True
        >>> is_camelcase("PascalCase")
        False
    """
    return isinstance(value, str) and bool(_CAMEL_RE.match(value))


def is_pascalcase(value: Any) -> bool:
    """Return True if the input string is in PascalCase.

    Args:
        value: The input to check.

    Returns:
        True if ``value`` is a string in PascalCase, False otherwise.

    Examples:
        >>> is_pascalcase("PascalCase")
        True
        >>> is_pascalcase("camelCase")
        False
    """
    return isinstance(value, str) and bool(_PASCAL_RE.match(value))


def is_kebabcase(value: Any) -> bool:
    """Return True if the input string is in kebab-case.

    Args:
        value: The input to check.

    Returns:
        True if ``value`` is a string in kebab-case, False otherwise.

    Examples:
        >>> is_kebabcase("kebab-case")
        True
        >>> is_kebabcase("snake_case")
        False
    """
    return isinstance(value, str) and bool(_KEBAB_RE.match(value))


def is_snakecase(value: Any) -> bool:
    """Return True if the input string is in snake_case.

    Args:
        value: The input to check.

    Returns:
        True if ``value`` is a string in snake_case, False otherwise.

    Examples:
        >>> is_snakecase("snake_case")
        True
        >>> is_snakecase("kebab-case")
        False
    """
    return isinstance(value, str) and bool(_SNAKE_RE.match(value))

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---
def pascalize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to PascalCase.

    Args:
        value: The string or collection to convert.

    Returns:
        The converted object with PascalCase keys or string.

    Examples:
        >>> pascalize("hello_world")
        "HelloWorld"
    """
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, pascalize)
    
    s     = str(value)
    words = _separate_words(s)
    return "".join(word.capitalize() for word in words)


def camelize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to camelCase.

    Args:
        value: The string or collection to convert.

    Returns:
        The converted object with camelCase keys or string.

    Examples:
        >>> camelize("hello_world")
        "helloWorld"
    """
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, camelize)

    s          = str(value)
    words      = _separate_words(s)
    pascalized = "".join(word.capitalize() for word in words)
    return pascalized[0].lower() + pascalized[1:] if pascalized else ""


def kebabize(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to kebab-case.

    Args:
        value: The string or collection to convert.

    Returns:
        The converted object with kebab-case keys or string.

    Examples:
        >>> kebabize("helloWorld")
        "hello-world"
    """
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, kebabize)

    s     = str(value)
    words = _separate_words(s)
    return "-".join(word.lower() for word in words)


def snakecase(value: Any) -> Any:
    """Convert a string, dictionary, or list of dictionaries to snake_case.

    Args:
        value: The string or collection to convert.

    Returns:
        The converted object with snake_case keys or string.

    Examples:
        >>> snakecase("helloWorld")
        "hello_world"
    """
    if isinstance(value, (list, Mapping)):
        return _process_keys(value, snakecase)

    s     = str(value)
    words = _separate_words(s)
    return "_".join(word.lower() for word in words)


# --- Aliases for backward compatibility ---
decamelize  = snakecase
depascalize = snakecase
dekebabize  = snakecase

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def _process_keys(data: Any, func: Callable) -> Any:
    """Recursively apply a function to dictionary keys or list elements.

    Args:
        data: The dictionary or list to process.
        func: The function to apply to keys.

    Returns:
        The processed data structure.
    """
    if isinstance(data, Mapping):
        return {func(k): _process_keys(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [_process_keys(i, func) for i in data]
    return data


def _separate_words(string: str) -> list[str]:
    """Split a string into words based on case and separators.

    Args:
        string: The input string to split.

    Returns:
        A list of words extracted from the string.
    """
    return _WORD_RE.findall(string)

# endregion
