#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Decorator.

This module provides generic decorators.
"""

from __future__ import annotations

__all__ = [
    "singleton",
]

import threading
from functools import wraps


# ==============================================================================
# region CREATION
# ==============================================================================

def singleton(cls):
    """Decorator to create a thread-safe singleton with no image caching."""
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
