#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic Collections.

This module provides generic collection containers with enhanced features.
"""

from __future__ import annotations

__all__ = [
    "DictList",
    "IndexList",
]

import inspect
from collections import defaultdict, UserDict, UserList
from typing import Any, Generic, Iterable, override, Type, TypeVar, Union


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class IndexList(UserList, Generic[T]):
    """A generic list that supports O(1) lookups by ``name`` (str) and
    optionally by ``id`` (int). Also supports optional strict type enforcement
    and auto-casting.

    This is equivalent to the following dictionary data structure:
    ::

        index_list: dict[str, DataClass] = {
            "key1": DataClass(name="key1", id=1, ...),
            "key2": DataClass(name="key2", id=2, ...),
            ...
        }
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        item_type: Type[T] | None = None,
        data: Iterable | None = None,
        key: str = "name",
        id: str | None = None,
    ):
        """Initialize a new instance.

        Args:
            item_type (Type[T], optional): The class type to enforce/cast.
                If None, it accepts any object. Defaults to None.
            data (Iterable, optional): Initial list of items. Defaults to None.
            key (str, optional): The attribute name to use as the lookup key.
                Defaults to 'name'.
            id (str, optional): The attribute name to use as the lookup id.
                If None, ID lookup is disabled. Defaults to None.
        """
        super().__init__()
        # Assign attributes
        self.item_type = item_type
        self.key_attr = key
        self.id_attr = id

        # 1. Try to infer item_type if missing and data exists
        if self.item_type is None and data:
            self.item_type = self._infer_type(data)

        # 2. Populate data (triggers validation if item_type exists)
        # Primary Index (Name) - Always active
        self._key_map: dict[str, T] = {}
        # Secondary Index (ID) - Active only if requested
        self._id_map: dict[int, T] | None = {} if id else None
        if data is not None:
            self.extend(data)

    # noinspection PyMethodMayBeStatic
    def _infer_type(self, data: Iterable) -> Type[T] | None:
        """Infer the item type from the first item in the iterable."""
        # Extract the first value from dict or kwargs
        first_val = data[0] if isinstance(data, list) and data else None

        # If we found a real object (not a primitive dict), return its type
        if first_val is not None and not isinstance(first_val, (dict, list, str, int, float)):
             return type(first_val)

        return None

    def _ensure_type(self, item: Any) -> T:
        """Validates or Casts the item."""
        # If no type is enforced, accept everything
        if self.item_type is None:
            return item

        # If it's already the correct type, pass it through
        if isinstance(item, self.item_type):
            return item

        # If it's a dict and the target, try to unpack it
        if isinstance(item, dict) and inspect.isclass(self.item_type):
            try:
                return self.item_type(**item)
            except TypeError as e:
                raise TypeError(
                    f"Cannot cast dict to {self.item_type.__name__}: {e}"
                )

        raise TypeError(
            f"Expected item of type '{self.item_type.__name__}' or a compatible "
            f"dict, but got {type(item).__name__}."
        )

    def _rebuild_indices(self):
        """Sync the O(1) lookup map with the list data."""
        self._key_map.clear()
        if self._id_map is not None:
            self._id_map.clear()

        for item in self.data:
            self._update_indices(item)

    def _update_indices(self, item: T):
        """Updates active indices for a single item."""
        # 1. Update name index
        if hasattr(item, self.key_attr):
            self._key_map[getattr(item, self.key_attr)] = item

        # 2. Update id nndex (only if enabled)
        if self._id_map is not None and hasattr(item, self.id_attr):
            self._id_map[getattr(item, self.id_attr)] = item

    # --- Representation ---
    @override
    def __repr__(self) -> str:
        type_name = self.item_type.__name__ if self.item_type else "Any"
        return f"<{self.__class__.__name__}[{type_name}] with {len(self)} items>"

    # --- Mathematical Operators ---
    @override
    def __add__(self, other) -> IndexList[T]:
        """Implement the addition operator (a + b)."""
        return self.__or__(other)

    def __or__(self, other) -> IndexList[T]:
        """Implement the union operator (a | b)."""
        if not isinstance(other, (UserList, list)):
             return NotImplemented

        new_obj = self.__class__(self.item_type, key=self.key_attr)
        new_obj.extend(self)
        new_obj.extend(other)
        return new_obj

    # --- Container / Sequence Methods ---
    @override
    def __getitem__(self, index: int | str | slice) -> Union[T, "IndexList[T]"]:
        """Return an item at the given ``index`` (or key).

        Support both list-style indexing and dictionary-style lookup by key:
            1. list[0]      --> Index lookup
            2. list["name"] --> Name lookup
        """
        if isinstance(index, str):
            # Dictionary behavior (Lookup by key)
            try:
                return self._key_map[index]
            except KeyError:
                raise KeyError(f"Item with {self.key_attr}='{index}' not found.")

        # Fallback to standard list behavior (int/slice)
        return super().__getitem__(index)

    @override
    def __setitem__(self, index: int, item: Any):
        """Define behavior for when an item is assigned to, using the notation
        self[key] = value.
        """
        # 1. Cast
        typed_item = self._ensure_type(item)

        # 2. Update data
        super().__setitem__(index, typed_item)

        # 3. Update index (Full rebuild needed because we don't know the old key)
        self._rebuild_indices()

    @override
    def __delitem__(self, index: int | slice):
        """Delete an item at the given ``index`` (or key)."""
        super().__delitem__(index)
        self._rebuild_indices()

    @override
    def __contains__(self, item: T | str | Any) -> bool:
        """Define behavior for membership tests using in and not in."""
        # Check by Key (O(1))
        if item in self._key_map:
            return True
        # Check by Object Identity (O(N))
        return super().__contains__(item)

    # --- Properties
    @property
    def keys(self) -> list[str]:
        """Return a list of keys."""
        return list(self._key_map.keys())

    @property
    def values(self) -> list[T]:
        """Return a list of values."""
        return list(self.data)

    # --- Retrieval ---
    def get(self, key: str | int, default: Any = None) -> Union[T, Any]:
        """Return the item associated with the given key.

        Supports retrieval by:
            1. get("name") -> Look in Key Map
            2. get(100)    -> Look in ID Map (IF enabled)
        """
        if isinstance(key, str):
            return self._key_map.get(key, default)

        elif isinstance(key, int):
            # Only check an ID map if the feature is enabled
            if self._id_map is not None:
                return self._id_map.get(key, default)
            # If ID lookup is disabled, an integer key means nothing here
            return default

        return default

    def with_id(self, value: int) -> T:
        """Explicitly retrieve by ID. Errors if feature disabled."""
        if self._id_map is None:
            raise NotImplementedError("ID lookup is not enabled for this list.")

        if value in self._id_map:
            return self._id_map[value]
        raise KeyError(f"Item with {self.id_attr}={value} not found.")

    # --- Mutation ---
    def apply(self, func):
        """Apply a function to all items in the list."""
        for item in self.data:
            self[item] = func(item)

    @override
    def append(self, item: Any):
        """Add a new item to the end of the list."""
        typed_item = self._ensure_type(item)
        super().append(typed_item)
        self._update_indices(typed_item)

    @override
    def extend(self, other: Iterable):
        """Add multiple items to the end of the list."""
        for item in other:
            self.append(item)

    @override
    def insert(self, i: int, item: Any):
        """Insert a new item at the given index."""
        typed_item = self._ensure_type(item)
        super().insert(i, typed_item)
        self._update_indices(typed_item)

    # --- Transformation ---
    def to_list(self) -> list[T]:
        """Convert back to a standard python list."""
        return list(self.data)

    def to_dict(self) -> dict[str, T]:
        """Convert back to a standard python dict."""
        return dict(self._key_map)

    def to_id_dict(self) -> dict[int, T]:
        """Convert back to a standard python dict."""
        return dict(self._id_map) if self._id_map else {}


class DictList(UserDict, Generic[K, V]):
    """Generic dictionary that stores multiple values under the same key.
    Also supports optional strict type enforcement and auto-casting.

    This is equivalent to the following dictionary data structure:
    ::

        dict_list: dict[str, list[Any]] = {
            "key1": [DataClass1(), DataClass1(), ...],
            "key2": [DataClass2(), DataClass2(), ...],
            ...
        }
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        item_type: Type[V] | None = None,
        data: dict[K, Union[V, list[V]]] | None = None,
        **kwargs
    ):
        """Initialize a new instance.

        Args:
            item_type (Type[V], optional): The class type to enforce/cast.
                If None, it tries to infer from ``data``. Defaults to None.
            data (dict[K, Union[V, list[V]]], optional): Initial dictionary to
                populate. Defaults to None.
            **kwargs: Additional key-value pairs to initialize.
        """
        super().__init__()
        # Assign attributes
        self.data = defaultdict(list)
        self.item_type = item_type

        # 1. Try to infer item_type if missing and data exists
        if self.item_type is None and data:
            self.item_type = self._infer_type(data)

        # 2. Populate data (triggers validation if item_type exists)
        if data is not None:
            self.update(data)
        if kwargs:
            self.update(kwargs)

    # noinspection PyMethodMayBeStatic
    def _infer_type(self, data: dict[K, Union[V, list[V]]]) -> Type[V] | None:
        """Infer the item type from the first item in the dictionary."""
        first_val = None

        # Extract the first value from dict or kwargs
        for v in data.values():
            if isinstance(v, list) and v:
                first_val = v[0]
                break
            elif v is not None:
                first_val = v
                break

        # If we found a real object (not a primitive dict), return its type
        if first_val is not None and not isinstance(first_val, (dict, list, str, int, float)):
             return type(first_val)

        return None

    def _ensure_type(self, item: Any) -> V:
        """Validate and cast item if ``item_type`` is set."""
        # If no type is enforced, accept everything
        if self.item_type is None:
            return item

        # If it's already the correct type, pass it through
        if isinstance(item, self.item_type):
            return item

        # If it's a dict and the target, try to unpack it
        if isinstance(item, dict) and inspect.isclass(self.item_type):
            try:
                return self.item_type(**item)
            except TypeError as e:
                raise TypeError(
                    f"Cannot to cast dict to {self.item_type.__name__}: {e}"
                )

        raise TypeError(
            f"Expected item of type '{self.item_type.__name__}' or a compatible "
            f"dict, but got {type(item).__name__}."
        )

    # --- Mathematical Operators ---
    def __add__(self, other) -> DictList[K, V]:
        """Implement the addition operator (a + b)."""
        return self.__or__(other)

    def __iadd__(self, other) -> DictList[K, V]:
        """Implement the in-place addition operator (a += b)."""
        return self.__ior__(other)

    def __or__(self, other) -> DictList[K, V]:
        """Implement the union operator (a | b)."""
        if not isinstance(other, (UserDict, dict)):
            return NotImplemented

        # 1. Create a new instance with the same item_type
        # We use type(self) to ensure subclasses (like BBoxList) work
        new_obj = type(self)(item_type=self.item_type)

        # 2. Add data from self (Left operand)
        new_obj.extend(self)

        # 3. Add data from other (Right operand)
        new_obj.extend(other)

        return new_obj

    def __ior__(self, other) -> DictList[K, V]:
        """Implement the in-place operator (a |= b)."""
        self.extend(other)
        return self

    # --- Container / Sequence Methods ---
    @override
    def __getitem__(self, key: K) -> list[V]:
        """Return an item at the given ``key``."""
        return self.data[key]

    @override
    def __setitem__(self, key: K, item: V | list[V]):
        """Define behavior for when an item is assigned to, using the notation
        self[key] = value.
        """
        if isinstance(item, list):
            self.data[key] = [self._ensure_type(i) for i in item]
        else:
            self.data[key] = [self._ensure_type(item)]

    # --- Properties ---
    @property
    def values_flat(self) -> list[V]:
        """Return a single list containing all items from all keys."""
        all_items = []
        for v in self.values():
            all_items.extend(v)
        return all_items

    @property
    def total_items(self) -> int:
        """Return the total number of items across all lists."""
        return sum(len(v) for v in self.data.values())

    # --- Creation ---
    @classmethod
    def from_keys(
        cls,
        keys: Iterable[str],
        item_type: Type[T] | None = None
    ) -> DictList[str, T]:
        """Create a new instance from a list of keys. Set all values to empty
        lists.
        """
        return cls(item_type=item_type, data={k: [] for k in keys})

    # --- Validation ---
    def verify(self):
        """Verify the integrity of the dictionary."""
        for k, v in self.data.items():
            if not isinstance(v, list):
                raise TypeError(
                    f"Expected list for key '{k}', but got {type(v).__name__}."
                )
            if (
                self.item_type is not None
                and any(not isinstance(item, self.item_type) for item in v)
            ):
                raise TypeError(
                    f"Expected all items in list '{k}' to be of type "
                    f"'{self.item_type.__name__}'."
                )

    # --- Mutation ---
    def apply(self, func):
        """Apply a function to all items in the dictionary."""
        for k in self.data:
            self.data[k] = [func(item) for item in self.data[k]]

    def append(self, key: K, item: V | list[V]):
        """Add a new item to the end of the list at the given key.

        Args:
            key (K): The key under which to append the item(s).
            item (V | list[V]): The item or list of items to append.
        """
        if isinstance(item, list):
            self.data[key].extend([self._ensure_type(i) for i in item])
        else:
            self.data[key].append(self._ensure_type(item))

    def extend(self, other: dict[K, list[V]]):
        """Add multiple items to the end of the list at the given key.

        Args:
            other (dict[K, list[V]]): A dictionary of key-list pairs to extend
                the current ``DictList`` with.
        """
        for k, v in other.items():
            self.append(k, v)

    # --- Transformation ---
    def to_dict(self) -> dict[K, list[V]]:
        """Convert back to a standard python dict."""
        return dict(self.data)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
