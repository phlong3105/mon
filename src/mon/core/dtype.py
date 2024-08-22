#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Types.

This module offer data handling capabilities, including lists, dictionaries,
tuples, sets, and more advanced data structures from the :obj:`collections`
module.

This name combines "data" and "types" to convey that the module provides various
data structures and types for data handling.
"""

from __future__ import annotations

__all__ = [
    "Enum",
    "concat_lists",
    "flatten_models_dict",
    "get_module_vars",
    "intersect_dicts",
    "intersect_ordered_dicts",
    "is_float",
    "is_int",
    "iter_to_iter",
    "iter_to_list",
    "iter_to_tuple",
    "shuffle_dict",
    "split_list",
    "to_1list",
    "to_1tuple",
    "to_2list",
    "to_2tuple",
    "to_3list",
    "to_3tuple",
    "to_4list",
    "to_4tuple",
    "to_5list",
    "to_5tuple",
    "to_6list",
    "to_6tuple",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_nlist",
    "to_ntuple",
    "to_pair",
    "to_quadruple",
    "to_single",
    "to_str",
    "to_triple",
    "to_tuple",
    "unique",
    "upcast",
]

import copy
import enum
import itertools
import random
import re
from collections import OrderedDict
from types import ModuleType
from typing import Any, Callable, Iterable

import numpy as np
import torch
from plum import dispatch


# region Enum

class Enum(enum.Enum):
    """An extension of Python :obj:`enum.Enum`."""
    
    @classmethod
    def random(cls):
        """Return a random enum."""
        return random.choice(seq=list(cls))
    
    @classmethod
    def random_value(cls):
        """Return a random enum value."""
        return cls.random().value
    
    @classmethod
    def keys(cls) -> list:
        """Return a list of all enums."""
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """A list of all enums' values."""
        return [e.value for e in cls]

# endregion


# region Collection

def intersect_dicts(x: dict, y: dict, exclude: list = []) -> dict:
    """Find the intersection between two :obj:`dict`.
    
    Args:
        x: The first :obj:`dict`.
        y: The second :obj:`dict`.
        exclude: A :obj:`list` of excluding keys. Default: ``[]``.
    
    Returns:
        A :obj:`dict` that contains only the keys that are in both :obj:`x`
        and :obj:`y`, and whose values are equal.
    """
    return {
        k: v for k, v in x.items()
        if k in y and not any(x in k for x in exclude) and v == y[k]
    }


def intersect_ordered_dicts(
    x      : OrderedDict,
    y      : OrderedDict,
    exclude: list = [],
) -> OrderedDict:
    """Find the intersection between two :obj:`OrderedDict`.
    
    Args:
        x: The first ordered :obj:`dict`.
        y: The second ordered :obj:`dict`.
        exclude: A :obj:`list` of excluding keys. Default: ``[]``.
    
    Returns:
        An :obj:`OrderedDict` that contains only the keys that are in both
        :obj:`x` and :obj:`y`, and whose values are equal.
    """
    return OrderedDict(
        (k, v) for k, v in x.items()
        if (k in y and not any(x in k for x in exclude) and v == y[k])
    )


def shuffle_dict(x: dict) -> dict:
    """Shuffle a :obj:`dict` randomly."""
    keys = list(x.keys())
    random.shuffle(keys)
    shuffled = {}
    for key in keys:
        shuffled[key] = x[key]
    return shuffled


def flatten_models_dict(x: dict) -> dict:
    """Flatten a nested dictionary of models into a single dictionary."""
    new_dict = {}
    for k1, v1 in x.items():
        for k2, v2 in v1.items():
            if isinstance(v2, dict):
                v2["arch"] = k1
            new_dict[k2] = v2
    return new_dict

# endregion


# region Module

def get_module_vars(module: ModuleType) -> dict:
    """Return all public variables of a module in a :obj:`dict`."""
    return {
        k: v for k, v in vars(module).items()
        if not (
            k == "__init__"
            or callable(k)
            or isinstance(v, ModuleType)
            or k.startswith(("_", "__", "annotations"))
        )
    }

# endregion


# region Numeric

def is_int(x: Any) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False
    
    
def is_float(x: Any) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def to_int(x: Any) -> int | None:
    """Convert a value to :obj:`int`."""
    if x is None:
        return None
    elif isinstance(x, str) and not is_int(x):
        raise ValueError(f"`x` must be a digit string, but got {x} ({type(x)}).")
    return int(x)


def to_float(x: Any) -> float | None:
    """Convert a value to :obj:`float`."""
    if x is None:
        return None
    elif isinstance(x, str) and not is_float(x):
        raise ValueError(f"`x` must be a digit string, but got {x} ({type(x)}).")
    return float(x)

# endregion


# region Parsing

def upcast(
    x        : torch.Tensor | np.ndarray,
    keep_type: bool = False
) -> torch.Tensor | np.ndarray:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.
    
    Args:
        x: An input of type :obj:`numpy.ndarray` or :obj:`torch.Tensor`.
        keep_type: If ``True``, keep the same type (int32  -> int64). Else
            upcast to a higher type (int32 -> float32).
            
    Return:
        A variable of higher type.
    """
    if x.dtype is torch.float16:
        return x.to(torch.float32)
    elif x.dtype is torch.float32:
        return x  # x.to(torch.float64)
    elif x.dtype is torch.int8:
        return x.to(torch.int16) if keep_type else x.to(torch.float16)
    elif x.dtype is torch.int16:
        return x.to(torch.int32) if keep_type else x.to(torch.float32)
    elif x.dtype is torch.int32:
        return x  # x.to(torch.int64) if keep_type else x.to(torch.float64)
    elif type(x) is np.float16:
        return x.astype(np.float32)
    elif type(x) is np.float32:
        return x  # x.astype(np.float64)
    elif type(x) is np.int16:
        return x.astype(np.int32) if keep_type else x.astype(np.float32)
    elif type(x) is np.int32:
        return x  # x.astype(np.int64) if keep_type else x.astype(np.int64)
    return x

# endregion


# region Sequence

def concat_lists(x: list[list]) -> list:
    """Concatenate a :obj:`list` of lists into a flattened :obj:`list`."""
    x = list(itertools.chain(*x))
    return x


def iter_to_iter(x: Iterable, item_type: type, return_type: type = None):
    """Convert an :obj:`Iterable` object to a desired sequence type specified
    by the :obj:`return_type`. Also, cast each item into the desired
    :obj:`item_type`.
    
    Args:
        x: An :obj:`Iterable` object.
        item_type: The item type.
        return_type: The desired iterable type. Default: ``None``.
    
    Returns:
        An :obj:`Iterable` object cast to the desired type.
    """
    if not isinstance(x, list | tuple | dict):
        raise TypeError(f"`x` must be a `list`, `tuple`, or `dict`, "
                        f"but got {type(x)}.")
    x = copy.deepcopy(x)
    x = map(item_type, x)
    if return_type is None:
        return x
    else:
        return return_type(x)


def iter_to_list(x: Iterable, item_type: type) -> list:
    """Convert an arbitrary :obj:`Iterable` object to a :obj:`list`."""
    return iter_to_iter(x=x, item_type=item_type, return_type=list)


def iter_to_tuple(x: Iterable, item_type: type) -> tuple:
    """Convert an arbitrary :obj:`Iterable` object to a :obj:`tuple`."""
    return iter_to_iter(x=x, item_type=item_type, return_type=tuple)


def split_list(x: list, n: int | list[int]) -> list[list]:
    """Slice a single :obj:`list` into a list of lists.
    
    Args:
        x: A :obj:`list` object.
        n: A number of sub-lists, or a :obj:`list` of integers to specify the
            length of each sub-list.
        
    Returns:
        A :obj:`list` of lists.
    
    Examples:
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> y = split_list(x, n=2)          # [1, 2, 3], [4, 5, 6]
        >>> z = split_list(x, n=[1, 3, 2])  # [1], [2, 3, 4], [5, 6]
    """
    if isinstance(n, int):
        if len(x) % n == 0:
            raise ValueError(f"`x` cannot be evenly split into {n} sub-list, "
                             f"but got length of `x` is {len(x)}.")
        n = [n] * int(len(x) / n)
    
    if sum(n) != len(x):
        raise ValueError(f"The total length of new sub-lists must match the "
                         f"length of `x`, but got {sum(n) != len(x)}.")
    
    y   = []
    idx = 0
    for i in range(len(n)):
        y.append(x[idx: idx + n[i]])
        idx += n[i]
    return y


def to_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list:
    """Convert an arbitrary value into a :obj:`list`.
    
    Args:
        x: An arbitrary value.
        sep: A :obj:`list` of delimiters to split a string.
    """
    if isinstance(x, list):
        x = x
    elif isinstance(x, tuple):
        x = list(x)
    elif isinstance(x, dict):
        x = list(x.values())
    elif isinstance(x, str):
        x = re.sub(r"^\s+|\s+$", "", x)
        x = re.sub(r"\s",        "", x)
        for s in sep:
            if s in x:
                x = x.split(s)
                break
        x = [x] if not isinstance(x, list) else x
    elif x is not None:
        x = [x]
        return x
    elif x is None:
        x = []
    return x


def to_int_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[int]:
    """Convert a string into a :obj:`list` of :obj:`int`."""
    x = to_list(x, sep=sep)
    x = [int(i) for i in x]
    return x


def to_float_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[float]:
    """Convert a string into a :obj:`list` of :obj:`float`."""
    x = to_list(x, sep=sep)
    x = [float(i) for i in x]
    return x


def to_nlist(n: int) -> Callable[[Any], list]:
    """Take an integer :obj:`n` and return a function that takes an
    :obj:`Iterable` object and returns a :obj:`list` of length :obj:`n`.
    
    Args:
        n: The number of elements in the :obj:`list`.
    
    Returns:
        A function that takes an integer and returns :obj:`list` of that
        integer :obj:`n` repeated n times.
    """
    def parse(x) -> list:
        if isinstance(x, Iterable):
            x = list(x)
            if len(x) == 1:
                x = list(itertools.repeat(x[0], n))
        else:
            x = list(itertools.repeat(x, n))
        return x
    
    return parse


to_1list = to_nlist(1)
to_2list = to_nlist(2)
to_3list = to_nlist(3)
to_4list = to_nlist(4)
to_5list = to_nlist(5)
to_6list = to_nlist(6)


def to_tuple(x: Any) -> tuple:
    """Convert an arbitrary value into a :obj:`tuple`."""
    if isinstance(x, list):
        x = tuple(x)
    elif isinstance(x, tuple):
        pass
    elif isinstance(x, dict):
        x = tuple(x.values())
    else:
        x = tuple(x)
    return x


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Take an integer :obj:`n` and return a function that takes an
    :obj:`Iterable` object and returns a :obj:`tuple` of length :obj:`n`.
    
    Args:
        n: The number of elements in the :obj:`tuple`.
    
    Returns:
        A function that takes an integer :obj:`n`n and returns a :obj:`tuple`
        of that integer repeated `n` times.
    """
    def parse(x) -> tuple:
        if isinstance(x, Iterable):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(itertools.repeat(x[0], n))
        else:
            x = tuple(itertools.repeat(x, n))
        return x
    
    return parse


to_1tuple    = to_ntuple(1)
to_2tuple    = to_ntuple(2)
to_3tuple    = to_ntuple(3)
to_4tuple    = to_ntuple(4)
to_5tuple    = to_ntuple(5)
to_6tuple    = to_ntuple(6)
to_single    = to_ntuple(1)
to_pair      = to_ntuple(2)
to_triple    = to_ntuple(3)
to_quadruple = to_ntuple(4)


@dispatch
def unique(x: list) -> list:
    """Get unique items from a :obj:`list`."""
    return list(set(x))


@dispatch
def unique(x: tuple) -> tuple:
    """Get unique items from a :obj:`tuple`."""
    return tuple(set(x))

# endregion


# region String

def to_str(x: Any, sep: str = ",") -> str:
    if isinstance(x, dict):
        x = [str(xi) for xi in x.values()]
    if not isinstance(x, list | tuple):
        x = [x]
    # Must be a list or tuple at this point
    x = [str(xi) for xi in x]
    if len(x) == 1:
        return f"{x[0]}"
    else:
        return sep.join(x)
    
# endregion
