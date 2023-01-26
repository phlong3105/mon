#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends Python builtin data types."""

from __future__ import annotations

__all__ = [
    "concat_lists", "copy_attr", "intersect_dicts", "intersect_ordered_dicts",
    "is_class", "iter_to_iter", "iter_to_list", "iter_to_tuple", "slice_list",
    "to_1tuple", "to_2tuple", "to_3tuple", "to_4tuple", "to_5tuple",
    "to_6tuple", "to_list", "to_ntuple", "to_tuple", "unique",
]

import copy
import inspect
import itertools
from collections import OrderedDict
from typing import Any, Callable, Iterable, Sequence

import multipledispatch


# region Object

def copy_attr(
    src    : object,
    dst    : object,
    include: Sequence = (),
    exclude: Sequence = (),
):
    """Copy all attributes from :param:`src` to :param:`dst`, except for those
    that start with an underscore, or are in the :param:`exclude` list.

    Args:
        src: An object to copy from.
        dst: An object to copy to.
        include: A list of attributes to include. If given, only attributes
            whose names reside in this list are copied.
        exclude: A list of attributes to exclude from copying.
    """
    for k, v in src.__dict__.items():
        if (len(include) and k not in include) or \
            k.startswith("_") or k in exclude:
            continue
        else:
            setattr(dst, k, v)


def is_class(input: Any) -> bool:
    return inspect.isclass(input)

# endregion


# region Collection

def intersect_dicts(da: dict, db: dict, exclude: Sequence = ()) -> dict:
    """Find the intersection between two dictionaries.
    
    Args:
        da: The first dictionary.
        db: The second dictionary.
        exclude: A list of excluding keys.
    
    Returns:
        A dictionary that contains only the keys that are in both dictionaries,
        and whose values are equal.
    """
    return {
        k: v for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v == db[k]
    }


def intersect_ordered_dicts(
    da     : OrderedDict,
    db     : OrderedDict,
    exclude: Sequence = (),
) -> OrderedDict:
    """Find the intersection between two ordered dictionaries.
    
    Args:
        da: The first ordered dictionary.
        db: The second ordered dictionary.
        exclude: A list of excluding keys.
    
    Returns:
        An ordered dictionary that contains only the keys that are in both
        dictionaries, and whose values are equal.
    """
    return OrderedDict(
        (k, v) for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and v == db[k])
    )

# endregion


# region Sequence

def concat_lists(ll: list[list]) -> list:
    """Concatenate a list of lists into a flattened list."""
    ll = ll.copy()
    return list(itertools.chain(*ll))


def iter_to_iter(
    inputs     : Iterable,
    item_type  : type,
    return_type: type | None = None,
):
    """Convert an iterable object to a desired sequence type specified by the
    :param:`return_type`. Also, casts each item into the desired
    :param:`item_type`.
    
    Args:
        inputs: The converting iterable object.
        item_type: The item type.
        return_type: The desired iterable type. Defaults to None.
    
    Returns:
        An iterable object cast to the desired type.
    """
    assert isinstance(inputs, Iterable)
    assert isinstance(item_type, type)
    inputs = copy.deepcopy(inputs)
    inputs = map(item_type, inputs)
    if return_type is None:
        return inputs
    else:
        return return_type(inputs)


def iter_to_list(inputs: Iterable, item_type: type) -> list:
    return iter_to_iter(inputs=inputs, item_type=item_type, return_type=list)


def iter_to_tuple(inputs: Iterable, item_type: type) -> tuple:
    return iter_to_iter(inputs=inputs, item_type=item_type, return_type=tuple)


def slice_list(input: list, lens: list[int]) -> list[list]:
    """Slice a single list into a list of lists.
    
    Args:
        input: A list.
        lens: A list of integers. Each item specifies the length of the
            corresponding sliced sub-list.
    
    Returns:
        A list of lists.
    """
    if isinstance(lens, int):
        assert len(input) % lens == 0
        lens = [lens] * int(len(input) / lens)
    
    assert isinstance(lens, list)
    assert len(lens) == len(input)
    
    out_list = []
    idx      = 0
    for i in range(len(lens)):
        out_list.append(input[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def to_list(input: Any) -> list:
    """Convert an arbitrary value into a list."""
    if isinstance(input, list):
        pass
    elif isinstance(input, tuple):
        input = list(input)
    elif isinstance(input, dict):
        input = [v for k, v in input.items()]
    else:
        input = [input]
    return input


def to_tuple(input: Any) -> tuple:
    """Convert an arbitrary value into a tuple."""
    if isinstance(input, list):
        input = tuple(input)
    elif isinstance(input, tuple):
        pass
    elif isinstance(input, dict):
        input = tuple([v for k, v in input.items()])
    else:
        input = tuple(input)
    return input


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Take an integer :param:`n` and return a function that takes an iterable
    object and returns a tuple of length :param:`n`.
    
    Args:
        n: The number of elements in the tuple.
    
    Returns:
        A function that takes an integer and returns a tuple of that integer
        repeated n times.
    """
    def parse(x) -> tuple:
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    
    return parse


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_5tuple = to_ntuple(5)
to_6tuple = to_ntuple(6)


@multipledispatch.dispatch(list)
def unique(input: list) -> list:
    """Get unique items from a list."""
    return list(set(input))


@multipledispatch.dispatch(tuple)
def unique(input: tuple) -> tuple:
    """Get unique items from a tuple."""
    return tuple(set(input))

# endregion
