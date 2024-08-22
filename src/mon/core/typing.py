#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Typing Module.

This module defines type alias used by PyTorch's layers. Here, we keep the
same naming convention as in :obj:`torch`.
"""

from __future__ import annotations

from typing import Callable, Optional, TypeVar, Union

from torch import nn

# Create some useful type aliases

# Template for arguments which can be supplied as a tuple, or which can be a
# scalar which PyTorch will internally broadcast to a tuple. Comes in several
# variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d
# operations.
T = TypeVar("T")
_scalar_or_tuple_any_t = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t   = Union[T, tuple[T]]
_scalar_or_tuple_2_t   = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t   = Union[T, tuple[T, T, T]]
_scalar_or_tuple_4_t   = Union[T, tuple[T, T, T, T]]
_scalar_or_tuple_5_t   = Union[T, tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t   = Union[T, tuple[T, T, T, T, T, T]]

# For arguments, which represent size parameters (for example kernel size,
# padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t   = _scalar_or_tuple_1_t[int]
_size_2_t   = _scalar_or_tuple_2_t[int]
_size_3_t   = _scalar_or_tuple_3_t[int]
_size_4_t   = _scalar_or_tuple_4_t[int]
_size_5_t   = _scalar_or_tuple_5_t[int]
_size_6_t   = _scalar_or_tuple_6_t[int]

# For arguments, which represent optional size parameters (for example adaptive
# pool parameters)
_size_any_opt_t = _scalar_or_tuple_any_t[Optional[int]]
_size_2_opt_t   = _scalar_or_tuple_2_t[Optional[int]]
_size_3_opt_t   = _scalar_or_tuple_3_t[Optional[int]]

# For arguments that represent a ratio to adjust each dimension of an input with
# (for example upsampling parameters)
_ratio_2_t   = _scalar_or_tuple_2_t[float]
_ratio_3_t   = _scalar_or_tuple_3_t[float]
_ratio_any_t = _scalar_or_tuple_any_t[float]

# Misc
_callable    = Union[None, nn.Module, Callable]
