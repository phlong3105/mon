#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for converting string case styles.

This module implements functions for converting strings between different case
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
    """Converts a string, dict, or list of dicts to pascal-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        str, dict, or list: The input converted to pascal-case.
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
    """Converts a string, dict, or list of dicts to camel-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
            
    Returns:
        str, dict, or list: The input converted to camel-case.
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
    """Converts a string, dict, or list of dicts to kebab-case.
   
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
            
    Returns:
        str, dict, or list: The input converted to kebab-case.
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
    """Converts a string, dict, or list of dicts to snake-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
            
    Returns:
        str, dict, or list: The input converted to snake-case.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, decamelize)

    s = _is_none(str_or_iter)
    if s.isupper() or s.isnumeric():
        return str_or_iter

    return _separate_words(_fix_abbreviations(s)).lower()


def depascalize(str_or_iter):
    """Converts a string, dict, or list of dicts to snake-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        str, dict, or list: The input converted to snake-case.
    """
    return decamelize(str_or_iter)


def dekebabize(str_or_iter):
    """Converts a string, dict, or list of dicts to snake-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        str, dict, or list: The input converted to snake-case.
    """
    if isinstance(str_or_iter, (list, Mapping)):
        return _process_keys(str_or_iter, dekebabize)

    s = _is_none(str_or_iter)
    if s.isnumeric():
        return str_or_iter

    return s.replace("-", "_")


snakecase = depascalize


# ----- Validation -----
def is_camelcase(str_or_iter) -> bool:
    """Checks if a string, dict, or list of dicts is camel-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
            
    Returns:
        bool: True if the input is camel-case, False otherwise.
    """
    return str_or_iter == camelize(str_or_iter)


def is_pascalcase(str_or_iter) -> bool:
    """Checks if a string, dict, or list of dicts is pascal-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        bool: True if the input is pascal-case, False otherwise.
    """
    return str_or_iter == pascalize(str_or_iter)


def is_kebabcase(str_or_iter) -> bool:
    """Checks if a string, dict, or list of dicts is kebab-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        bool: True if the input is kebab-case, False otherwise.
    """
    return str_or_iter == kebabize(str_or_iter)


def is_snakecase(str_or_iter) -> bool:
    """Checks if a string, dict, or list of dicts is snake-case.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
    
    Returns:
        bool: True if the input is snake-case, False otherwise.
    """
    if is_kebabcase(str_or_iter) and not is_camelcase(str_or_iter):
        return False

    return str_or_iter == decamelize(str_or_iter)


def _is_none(_in) -> str:
    """Determines if the input is None, returning an empty string if so.
    
    Returns:
        str: An empty string if the input is None; otherwise, the input
            converted to a string with all whitespace removed.
    """
    return "" if _in is None else re.sub(r"\s+", "", str(_in))


# ----- Utils -----
def _process_keys(str_or_iter, fn):
    """Recursively process keys in a dict or list using a specified function.
    
    Args:
        str_or_iter (str, dict, or list): Input string, dictionary, or list of
            dictionaries.
        fn (callable): Function to apply to each key.
    
    Returns:
        str, dict, or list: The input with keys processed by the specified function.
    """
    if isinstance(str_or_iter, list):
        return [_process_keys(k, fn) for k in str_or_iter]
    if isinstance(str_or_iter, Mapping):
        return {fn(k): _process_keys(v, fn) for k, v in str_or_iter.items()}
    return str_or_iter


def _fix_abbreviations(string: str) -> str:
    """Rewrites incorrectly cased acronyms, initialisms, and abbreviations,
    allowing them to be decamelized correctly. For example, given the string
    "APIResponse", this function is responsible for ensuring the output is
    "api_response" instead of "a_p_i_response".
    
    Args:
        string (str): A string that may contain an incorrectly cased abbreviation.
    
    Returns:
        str: A rewritten string with properly cased abbreviations.
    """
    return ACRONYM_RE.sub(lambda m: m.group(0).title(), string)


def _separate_words(string: str, separator: str = "_") -> str:
    """Splits words that are separated by case differentiation.
    
    Args:
        string (str): A string that may contain an incorrectly cased abbreviation.
        separator (str): A string used to separate the words. Defaults to "_".
    
    Returns:
        str: A string with words separated by the specified separator.
    """
    return separator.join(s for s in SPLIT_RE.split(string) if s)
