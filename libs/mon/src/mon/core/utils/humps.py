#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""String case conversion and validation utilities.

This module provides recursive converters and validators for camelCase,
PascalCase, kebab-case, and snake_case, supporting structured data.
"""

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


# ==============================================================================
# VALIDATION & SANITIZATION (Integrity Checks)
# ==============================================================================

# --- Verify ---
def is_camelcase(str_or_iter) -> bool:
    """Return True if input is camelCase.

    Args:
        str_or_iter: String or structure to validate.

    Returns:
        True if the input equals its camelized form.
    """
    return str_or_iter == camelize(str_or_iter)


def is_pascalcase(str_or_iter) -> bool:
    """Return True if input is PascalCase.

    Args:
        str_or_iter: String or structure to validate.

    Returns:
        True if the input equals its pascalized form.
    """
    return str_or_iter == pascalize(str_or_iter)


def is_kebabcase(str_or_iter) -> bool:
    """Return True if input is kebab-case.

    Args:
        str_or_iter: String or structure to validate.

    Returns:
        True if the input equals its kebabized form.
    """
    return str_or_iter == kebabize(str_or_iter)


def is_snakecase(str_or_iter) -> bool:
    """Return True if input is snake_case.

    Args:
        str_or_iter: String or structure to validate.

    Returns:
        True if the input equals its decamelized form. Treat certain kebab-case
        inputs as non-snake to avoid false positives.
    """
    if is_kebabcase(str_or_iter) and not is_camelcase(str_or_iter):
        return False

    return str_or_iter == decamelize(str_or_iter)


def _is_none(_in) -> str:
    """Normalize None to an empty string and collapse whitespace.

    Args:
        _in: Input value.

    Returns:
        Compact string with internal whitespace removed, or empty string for None.
    """
    return "" if _in is None else re.sub(r"\s+", "", str(_in))


# ==============================================================================
# CASE TRANSFORMATION
# ==============================================================================

# --- Structural Converters ---
def pascalize(str_or_iter):
    """Convert input to PascalCase.

    Args:
        str_or_iter: A string, mapping, or list. If a mapping or list is
            provided, convert keys/elements recursively.

    Returns:
        Converted value in PascalCase or a structure with converted keys.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, pascalize)

    s = _is_none(str_or_iter)
    if s.isupper() or s.isnumeric():
        return str_or_iter

    def _replace_fn(match):
        return match.group(1)[0].upper() + match.group(1)[1:]

    s = camelize(PASCAL_RE.sub(_replace_fn, s))
    return s[0].upper() + s[1:] if len(s) != 0 else s


def camelize(str_or_iter):
    """Convert input to camelCase.

    Args:
        str_or_iter: A string, mapping, or list. If a mapping or list is
            provided, convert keys/elements recursively.

    Returns:
        Converted value in camelCase or a structure with converted keys.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, camelize)

    s = _is_none(str_or_iter)
    if s.isupper() or s.isnumeric():
        return str_or_iter

    if len(s) != 0 and not s[:2].isupper():
        s = s[0].lower() + s[1:]

    # For string "hello_world", match will contain
    #             the regex capture group for "_w".
    return UNDERSCORE_RE.sub(lambda m: m.group(0)[-1].upper(), s)


def kebabize(str_or_iter):
    """Convert input to kebab-case.

    Args:
        str_or_iter: A string, mapping, or list. If a mapping or list is
            provided, convert keys/elements recursively.

    Returns:
        Converted value in kebab-case or a structure with converted keys.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, kebabize)

    s = _is_none(str_or_iter)
    if s.isnumeric():
        return str_or_iter

    if not (s.isupper()) and (is_camelcase(s) or is_pascalcase(s)):
        return (
            _separate_words(
                string=_fix_abbreviations(s),
                separator="-"
            ).lower()
        )

    return UNDERSCORE_RE.sub(lambda m: "-" + m.group(0)[-1], s)


def decamelize(str_or_iter):
    """Convert input to snake_case.

    Args:
        str_or_iter: A string, mapping, or list. If a mapping or list is
            provided, convert keys/elements recursively.

    Returns:
        Converted value in snake_case or a structure with converted keys.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, decamelize)

    s = _is_none(str_or_iter)
    if s.isupper() or s.isnumeric():
        return str_or_iter

    return _separate_words(_fix_abbreviations(s)).lower()


def depascalize(str_or_iter):
    """Alias for decamelize; convert input to snake_case.

    Args:
        str_or_iter: Input to convert; behaves like decamelize.

    Returns:
        Converted value in snake_case.
    """
    return decamelize(str_or_iter)


def dekebabize(str_or_iter):
    """Convert kebab-case input to snake_case.

    Args:
        str_or_iter: A string, mapping, or list. If a mapping or list is
            provided, convert keys/elements recursively.

    Returns:
        Converted value in snake_case or a structure with converted keys.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, dekebabize)

    s = _is_none(str_or_iter)
    if s.isnumeric():
        return str_or_iter

    return s.replace("-", "_")


snakecase = depascalize


# ==============================================================================
# INTERNALS & RECURSION LOGIC
# ==============================================================================

# --- Regex Patterns ---
ACRONYM_RE    = re.compile(r"([A-Z\d]+)(?=[A-Z\d]|$)")
PASCAL_RE     = re.compile(r"([^\-_]+)")
SPLIT_RE      = re.compile(r"([\-_]*[A-Z][^A-Z]*[\-_]*)")
UNDERSCORE_RE = re.compile(r"(?<=[^\-_])[\-_]+[^\-_]")


# --- Structural Walkers --
def _process_keys(str_or_iter, fn):
    """Recursively apply a conversion function to mapping keys or list elements.

    Args:
        str_or_iter: Input mapping or list to process.
        fn: Conversion function to apply to keys/strings.

    Returns:
        Processed structure with fn applied to keys or elements.
    """
    if isinstance(str_or_iter, list):
        return [_process_keys(k, fn) for k in str_or_iter]
    if isinstance(str_or_iter, Mapping):
        return {fn(k): _process_keys(v, fn) for k, v in str_or_iter.items()}
    return str_or_iter


def _fix_abbreviations(string: str) -> str:
    """Normalize acronym capitalization for consistent splitting.

    Args:
        string: Input string possibly containing uppercase acronyms.

    Returns:
        String with acronyms title-cased for consistent splitting.
    """
    return ACRONYM_RE.sub(lambda m: m.group(0).title(), string)


def _separate_words(string: str, separator: str = "_") -> str:
    """Split camel or Pascal strings into words and join with a separator.

    Args:
        string: Input camel or Pascal string.
        separator: Separator to join the extracted words.

    Returns:
        String with words joined by the given separator.
    """
    return separator.join(s for s in SPLIT_RE.split(string) if s)
