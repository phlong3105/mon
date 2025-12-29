#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mixins for metadata operations.

This module provides mixins that return information about the data containers
without changing it.
"""

__all__ = [
    "RegistrableMixin",
]

import abc

from mon.core import Task


# ==============================================================================
# GLOBAL CONFIGURATIONS (Constants)
# ==============================================================================

# --- Constants (Global defaults, versioning) ---


# --- Environment ---


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---
class RegistrableMixin(abc.ABC):
    """A mixin class that adds metadata attribute to data containers for
    factory registration purposes.

    Define common dataset attributes for categorization, such as supported tasks.
    This is useful for factory-related operations.

    Attributes:
        _tasks (list[Task]): A list of supported tasks. Defaults to an empty
            list and should be overridden in subclasses.
    """

    _tasks: list[Task] = []

    # --- Properties ---
    @property
    def tasks(self) -> list[Task]:
        """Return the list of supported tasks."""
        return self._tasks


# --- Resolve (Retrieving spokes by name/key) ---


# ==============================================================================
# METADATA EXTRACTION (EXIF/Header Parsing)
# ==============================================================================

# --- Parse (EXIF, Header, and Sidecar parsing) ---
