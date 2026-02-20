#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Singleton Decorator and Metaclass.

This module provides two ways to create singleton classes in Python:
    1. A decorator-based approach using the `singleton` function.
    2. A metaclass-based approach using the `SingletonMeta` class.
"""

from __future__ import annotations

__all__ = [
    "SingletonMeta",
    "singleton",
]

import threading
from functools import wraps


# ==============================================================================
# region CREATION
# ==============================================================================

def singleton(cls):
    """Decorator to create a thread-safe singleton with no image caching.

    Examples:
        >>> @singleton
        ... class DatabaseConnection: pass
    """

    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                # Creates the instance only if it doesn't exist
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class SingletonMeta(type):
    """A thread-safe implementation of Singleton using a metaclass.

    Examples:
        >>> class DatabaseConnection(metaclass=SingletonMeta): pass
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check (no lock) for high performance
        if cls not in cls._instances:
            # Only lock when the instance might need to be created
            with cls._lock:
                # Second check (with lock) to prevent race conditions
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
