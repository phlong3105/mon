#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Singleton.

This module provides singleton utilities.
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
