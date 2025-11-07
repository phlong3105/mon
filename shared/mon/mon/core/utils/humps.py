#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functions for converting strings between different case
styles, including camel-case, pascal-case, kebab-case, and snake-case. It also
includes functions to validate if a string is in a specific case style.
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


ACRONYM_RE    = re.compile(r"([A-Z\d]+)(?=[A-Z\d]|$)")
PASCAL_RE     = re.compile(r"([^\-_]+)")
SPLIT_RE      = re.compile(r"([\-_]*[A-Z][^A-Z]*[\-_]*)")
UNDERSCORE_RE = re.compile(r"(?<=[^\-_])[\-_]+[^\-_]")


# ----- Convert -----
def pascalize(str_or_iter):
    """Convert a ``str``, ``dict``, or ``list`` of dicts to pascal-case."""
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
    """Convert a ``str``, ``dict``, or ``list`` of dicts to camel-case."""
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
    """Convert a ``str``, ``dict``, or ``list`` of dicts to kebab-case."""
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
    """Convert a ``str``, ``dict``, or ``list`` of dicts to snake-case."""
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, decamelize)

    s = _is_none(str_or_iter)
    if s.isupper() or s.isnumeric():
        return str_or_iter

    return _separate_words(_fix_abbreviations(s)).lower()


def depascalize(str_or_iter):
    """Convert a ``str``, ``dict``, or ``list`` of dicts to snake-case."""
    return decamelize(str_or_iter)


def dekebabize(str_or_iter):
    """Convert a ``str``, ``dict``, or ``list`` of dicts to snake-case."""
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, dekebabize)

    s = _is_none(str_or_iter)
    if s.isnumeric():
        return str_or_iter

    return s.replace("-", "_")


snakecase = depascalize


# ----- Validation -----
def is_camelcase(str_or_iter) -> bool:
    """Determine if a ``str``, ``dict``, or ``list`` of dicts is camel-case."""
    return str_or_iter == camelize(str_or_iter)


def is_pascalcase(str_or_iter) -> bool:
    """Determine if a ``str``, ``dict``, or ``list`` of dicts is pascal-case."""
    return str_or_iter == pascalize(str_or_iter)


def is_kebabcase(str_or_iter) -> bool:
    """Determine if a ``str``, ``dict``, or ``list`` of dicts is camel-case."""
    return str_or_iter == kebabize(str_or_iter)


def is_snakecase(str_or_iter) -> bool:
    """Determine if a ``str``, ``dict``, or ``list`` of dicts is snake-case."""
    if is_kebabcase(str_or_iter) and not is_camelcase(str_or_iter):
        return False

    return str_or_iter == decamelize(str_or_iter)


def _is_none(_in) -> str:
    """Determine if the input is ``None`` and returns a ``str`` with white-space
    removed.
    
    Returns:
        An empty sting if ``_in`` is ``None``, else the input is returned with
        white-space removed.
    """
    return "" if _in is None else re.sub(r"\s+", "", str(_in))


# ----- Utils -----
def _process_keys(str_or_iter, fn):
    if isinstance(str_or_iter, list):
        return [_process_keys(k, fn) for k in str_or_iter]
    if isinstance(str_or_iter, Mapping):
        return {fn(k): _process_keys(v, fn) for k, v in str_or_iter.items()}
    return str_or_iter


def _fix_abbreviations(string: str) -> str:
    """Rewrite incorrectly cased acronyms, initialisms, and abbreviations,
    allowing them to be decamelized correctly. For example, given the string
    "APIResponse", this function is responsible for ensuring the output is
    "api_response" instead of "a_p_i_response".
    
    Args:
        string: A string that may contain an incorrectly cased abbreviation.
    
    Returns:
        A rewritten ``str`` that is safe for decamelization.
    """
    return ACRONYM_RE.sub(lambda m: m.group(0).title(), string)


def _separate_words(string: str, separator: str = "_") -> str:
    """Split words that are separated by case differentiation.
    
    Args:
        string: Original string to be split.
        separator: String by which the individual words will be put back together.
    
    Returns:
        A ``str`` with words separated by the specified separator.
    """
    return separator.join(s for s in SPLIT_RE.split(string) if s)
