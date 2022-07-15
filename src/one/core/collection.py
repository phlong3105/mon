#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collections such as: list, tuple, dict.
"""

from __future__ import annotations

import collections
import inspect
import itertools
import sys
from collections import abc
from collections import OrderedDict
from typing import Iterable
from typing import Sequence
from typing import Union

from multipledispatch import dispatch

from one.core.globals import Int2Or3T
from one.core.globals import Int2T


# MARK: - Functional

def concat_lists(ll: list[list]) -> list:
    """Concatenate a list of list into a single list.
    
    Args:
        ll (list[list]):
            A list of list.
    
    Returns:
        (list):
            Concatenated list.
    """
    return list(itertools.chain(*ll))


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, options to only include [...] and to
    exclude [...].
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or \
            k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(
    da     : dict,
    db     : dict,
    exclude: Union[tuple, list] = ()
) -> dict:
    """Dictionary intersection omitting `exclude` keys, using da values.
    
    Args:
        da (dict):
            Dict a.
        db (dict):
            Dict b.
        exclude (tuple, list):
            Exclude keys.
        
    Returns:
        (dict):
            Dictionary intersection.
    """
    return {
        k: v for k, v in da.items()
        if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape
    }


def intersect_ordered_dicts(
    da     : OrderedDict,
    db     : OrderedDict,
    exclude: Union[tuple, list] = ()
) -> OrderedDict:
    """Dictionary intersection omitting `exclude` keys, using da values.
    
    Args:
        da (dict):
            Dict a.
        db (dict):
            Dict b.
        exclude (tuple, list):
            Exclude keys.
        
    Returns:
        (dict):
            Dictionary intersection.
    """
    return OrderedDict(
        (k, v) for k, v in da.items()
        if (k in db and not any(x in k for x in exclude) and v.shape == db[k].shape)
    )


def is_seq_of(
    s        : Sequence,
    item_type: type,
    seq_type : Union[type, None] = None
) -> bool:
    """Check whether `s` is a sequence of some type.
    
    Args:
        s (Sequence):
            Sequence to be checked.
        item_type (type):
            Expected type of sequence items.
        seq_type (type, None):
            Expected sequence type. Default: `None`.
    
    Return:
        (bool):
            `True` if `s` is a sequence of type `seq_type` containing item of
            type `item_type`. Else `False`.
    """
    if seq_type is None:
        seq_type = abc.Sequence
    if not isinstance(seq_type, type):
        raise TypeError(f"`seq_type` must be a valid type. But got: {seq_type}.")
    
    if not isinstance(s, seq_type):
        return False
    for item in s:
        if not isinstance(item, item_type):
            return False
    return True


def is_dict_of(d: dict, item_type: type) -> bool:
    """Check whether `d` is a dictionary of some type.
    
    Args:
        d (dict):
            Dictionary to be checked.
        item_type (type):
            Expected type of items.
    
    Return:
        (bool):
            `True` if `s` is a dictionary containing item of type `item_type`.
            Else `False`.
    """
    if not isinstance(item_type, type):
        raise TypeError(f"`item_type` must be a valid type. But got: {item_type}.")

    return all(isinstance(v, item_type) for k, v in d.items())


def is_list_of(l: list, item_type: type) -> bool:
    """Check whether `l` is a list of some type.
    
    Args:
        l (list):
            List to be checked.
        item_type (type):
            Expected type of items.
    
    Return:
        (bool):
            `True` if `s` is a list containing item of type `item_type`.
            Else `False`.
    """
    return is_seq_of(s=l, item_type=item_type, seq_type=list)


def is_tuple_of(t: tuple, item_type: type) -> bool:
    """Check whether `t` is a tuple of some type.
    
    Args:
        t (tuple):
            Dictionary to be checked.
        item_type (type):
            Expected type of items.
    
    Return:
        (bool):
            `True` if `s` is a tuple containing item of type `item_type`.
            Else `False`.
    """
    return is_seq_of(s=t, item_type=item_type, seq_type=tuple)


def slice_list(l: list, lens: Union[int, list[int]]) -> list[list]:
    """Slice a list into several sub-lists of various lengths.
    
    Args:
        l (list):
            List to be sliced.
        lens (int, list[int]):
            Expected length of each output sub-list. If `int`, split to equal
            length. If `list[int]`, split each sub-list to dedicated length.
    
    Returns:
        out_list (list):
            A list of sliced lists.
    """
    if isinstance(lens, int):
        if len(l) % lens != 0:
            raise ValueError(f"Length of `l` must be divisible by `lens`. "
                             f"But got: {len(l)} % {lens} != 0.")
        lens = [lens] * int(len(l) / lens)
    
    if not isinstance(lens, list):
        raise TypeError(f"`indices` must be an `int` or `list[int]`. "
                        f"But got: {type(lens)}.")
    if sum(lens) != len(l):
        raise ValueError(f"Sum of `lens` and length of `l` must be the same. "
                         f"But got: {sum(lens)} != {len(l)}.")
    
    out_list = []
    idx      = 0
    for i in range(len(lens)):
        out_list.append(l[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def to_iter(
    inputs     : Iterable,
    item_type  : type,
    return_type: Union[type, None] = None
):
    """Cast items of an iterable object into some type.
    
    Args:
        inputs (Iterable):
            Iterable object.
        item_type (type):
            Item type.
        return_type (type, None):
            If specified, the iterable object will be converted to this type,
            otherwise an iterator. Default: `None`.
            
    Returns:
        Iterable object of type `return_type` containing items of type
        `item_type`.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError(f"`inputs` must be an iterable object. "
                        f"But got: {type(inputs)}.")
    if not isinstance(item_type, type):
        raise TypeError(f"`item_type` must be a valid type. "
                        f"But got {type(item_type)}.")

    out_iterable = map(item_type, inputs)
    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def to_list(inputs: Iterable, item_type: type) -> list:
    """Cast items of an iterable object into a list of some type.

    Args:
        inputs (Iterable):
            Iterable object.
        item_type (type):
            Item type.
    
    Returns:
        List containing items of type `item_type`.
    """
    return to_iter(inputs=inputs, item_type=item_type, return_type=list)


def to_ntuple(n: int) -> tuple:
    """A helper functions to cast input to n-tuple."""
    
    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    
    return parse


def to_size(size: Int2Or3T) -> Int2T:
    """Cast size object of any format into standard [H, W].
    
    Args:
        size (Int2Or3T):
            Size object of any format.
            
    Returns:
        size (Int2T):
            Size of [H, W].
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:
            size = size[0:2]
        if len(size) == 1:
            size = (size[0], size[0])
    elif isinstance(size, (int, float)):
        size = (size, size)
    return tuple(size)


def to_tuple(inputs: Iterable, item_type: type):
    """Cast items of an iterable object into a tuple of some type.

    Args:
        inputs (Iterable):
            Iterable object.
        item_type (type):
            Item type.
    
    Returns:
        Tuple containing items of type `item_type`.
    """
    return to_iter(inputs=inputs, item_type=item_type, return_type=tuple)


@dispatch(list)
def unique(l: list) -> list:
    """Return a list with only unique items.
    
    Args:
        l (list):
            List that may contain duplicate items.
    
    Returns:
        l (list):
            List containing only unique items.
    """
    return list(set(l))


@dispatch(tuple)
def unique(t: tuple) -> tuple:
    """Return a tuple with only unique items.
    
    Args:
        t (tuple):
            List that may contain duplicate items.
    
    Returns:
        t (tuple):
            List containing only unique items.
    """
    return tuple(set(t))


# MARK: - Alias

to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_5tuple = to_ntuple(5)
to_6tuple = to_ntuple(6)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
