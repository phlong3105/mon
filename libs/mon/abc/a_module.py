#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Magic methods in a class.

A comprehensive boilerplate for Python classes implementing core dunder methods.
"""

from __future__ import annotations

from typing import Any, Iterator


class Foo:
    """A template class demonstrating the most common Python dunder methods."""
    
    # --- Lifecycle & Initialization ---
    def __new__(cls, *args, **kwargs) -> Foo:
        """Called to create a new instance of the class."""
        instance = super().__new__(cls)
        return instance
    
    def __init__(self, value: Any = None):
        """Initialize a new instance."""
        self.value = value
        self._data = []  # Internal storage for container methods
    
    def __del__(self):
        """Finalizer called when the object is about to be destroyed."""
        pass
    
    # --- Comparison Operators ---
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return self.value == other.value
    
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return not self == other
    
    def __lt__(self, other: Foo) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return self.value < other.value
    
    def __gt__(self, other: Foo) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return self.value > other.value
    
    def __le__(self, other: Foo) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return self.value <= other.value
    
    def __ge__(self, other: Foo) -> bool:
        if not isinstance(other, Foo):
            return NotImplemented
        return self.value >= other.value
    
    # --- Representation ---
    def __str__(self) -> str:
        """Informal string representation for end-users (print)."""
        return f"Foo with value: {self.value}"
    
    def __repr__(self) -> str:
        """Official string representation for developers (eval-able)."""
        return f"{self.__class__.__name__}(value={self.value!r})"
    
    def __format__(self, format_spec: str) -> str:
        """Custom behavior for f-string formatting."""
        return format(str(self.value), format_spec)
    
    def __hash__(self) -> int:
        """Allow the object to be used as a key in a dictionary or in a set."""
        return hash((self.__class__, self.value))
    
    # --- Mathematical Operators ---
    def __add__(self, other: Foo) -> Foo:
        return Foo(self.value + other.value)
    
    def __sub__(self, other: Foo) -> Foo:
        return Foo(self.value - other.value)
    
    # --- Type Conversion ---
    def __bool__(self) -> bool:
        return bool(self.value)
    
    def __int__(self) -> int:
        return int(self.value)
    
    # --- Attribute Access ---
    def __getattr__(self, name: str) -> Any:
        """Called only if the attribute was not found in the usual places."""
        return f"Attribute {name} not found"
    
    def __setattr__(self, name: str, value: Any):
        """Intercept every attribute assignment."""
        super().__setattr__(name, value)
    
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self._data)
    
    def __getitem__(self, index: int) -> Any:
        """Define behavior for when an item is accessed, using the notation
        self[key].
        """
        return self._data[index]
    
    def __setitem__(self, index: int, value: Any):
        """Define behavior for when an item is assigned to, using the notation
        self[key] = value.
        """
        self._data[index] = value
    
    def __iter__(self) -> Iterator:
        """Return an iterator for the container."""
        return iter(self._data)
    
    def __contains__(self, item: Any) -> bool:
        """Define behavior for membership tests using in and not in."""
        return item in self._data
    
    # --- Callable & Context Manager ---
    def __call__(self, *args, **kwargs) -> Any:
        """Allow the instance to be called like a function: foo()."""
        print("Foo instance was called!")
        return self.value
    
    def __enter__(self) -> Foo:
        """Setup for 'with' statement."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown for 'with' statement."""
        pass
