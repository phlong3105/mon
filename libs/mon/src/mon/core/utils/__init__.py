#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of string case and type utility functions.

This package contains helpers for string case conversion, type coercion,
collection utilities, and dictionary operations.
"""

__all__ = [
    "camelize",
    "create_combinations",
    "decamelize",
    "dekebabize",
    "depascalize",
    "is_camelcase",
    "is_float",
    "is_int",
    "is_kebabcase",
    "is_pascalcase",
    "is_snakecase",
    "kebabize",
    "merge_dicts",
    "pascalize",
    "snakecase",
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

from .humps import (
    camelize,
    decamelize,
    dekebabize,
    depascalize,
    is_camelcase,
    is_kebabcase,
    is_pascalcase,
    is_snakecase,
    kebabize,
    pascalize,
    snakecase,
)
from .types_extension import (
    create_combinations,
    is_float,
    is_int,
    merge_dicts,
    sort,
    to_float,
    to_float_list,
    to_int,
    to_int_list,
    to_list,
    to_ntuple,
    to_str,
    unique,
)
