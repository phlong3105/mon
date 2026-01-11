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
        _name (str): The name of the data container. Must be overridden in
            subclasses.
        _tasks (list[Task]): A list of supported tasks. Must be overridden in
            subclasses.
    """
    
    _name : str        = None
    _tasks: list[Task] = []
    
    # --- Lifecycle & Initialization ---
    def __init__(self, name: str = None, tasks: list[Task] = None, *args, **kwargs):
        """Initialize a new instance.
        
        Args:
            name: The name of the data container. If provided, it overrides the
                class-level default.
            tasks: A list of supported tasks. If provided, it overrides the
                class-level default.
        """
        # If provided, these instance variables will override the class-level defaults
        if name is not None:
            self._name = name
        if tasks is not None:
            # We use list() to create a copy, preventing shared state bugs
            self._tasks = list(tasks)
        
        # Continue the initialization chain
        super().__init__(*args, **kwargs)
        
    def __init_subclass__(cls, *args, **kwargs):
        """Called when inheriting from this class."""
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_name", "_tasks"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must explicitly define class "
                                f"attribute '{attr}' of type str or list[Task].")
        
        # Check for VALID values
        if cls._name is None:
            raise ValueError(f"Expected '{cls.__name__}' to define the '_name' attribute.")
        if not cls._tasks:  # Checks for None, empty list [], or empty tuple ()
            raise ValueError(f"Expected '{cls.__name__}' to define at least one "
                             f"supported task in the '_tasks' attribute.")
        
    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the name of the data container."""
        return self._name
    
    @property
    def tasks(self) -> list[Task]:
        """Return the list of supported tasks."""
        return self._tasks


# --- Resolve (Retrieving spokes by name/key) ---


# ==============================================================================
# METADATA EXTRACTION (EXIF/Header Parsing)
# ==============================================================================

# --- Parse (EXIF, Header, and Sidecar parsing) ---
