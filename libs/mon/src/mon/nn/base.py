#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for neural networks."""

from __future__ import annotations

__all__ = [
    "Container",
    "Module",
    "ModuleDict",
    "ModuleList",
    "ParameterDict",
    "ParameterList",
    "RegistrableMixin",
    "Sequential",
]

from torch.nn.modules.container import (
    Container,
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from torch.nn.modules.module import Module

from mon.core import MLType, Path, Task


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---

class RegistrableMixin:
    """A mixin class that adds metadata attribute to deep learning models for
    factory registration purposes.

    Attributes:
        arch: The model's architecture. `Must be defined in subclasses or set
            during initialization.`
        name: The model's name. `Must be defined in subclasses and set during
            initialization.`
        tasks: A list of tasks that the model can perform. `Must be defined in
            subclasses or set during initialization.`
        mltypes: A list of learning types that the model can perform. `Must be
            defined in subclasses or set during initialization.`
        model_dir: The model's directory. `Must be defined in subclasses or
            set during initialization.`
    """

    arch     : str          = ""
    name     : str          = ""
    tasks    : list[Task]   = []
    mltypes  : list[MLType] = []
    model_dir: Path         = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        arch     : str          | None = None,
        name     : str          | None = None,
        tasks    : list[Task]   | None = None,
        mltypes  : list[MLType] | None = None,
        model_dir: Path         | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            arch: Architecture of the data container. If provided, it overrides
                class-level default. Defaults to None.
            name: Name of the data container. If provided, it overrides the
                class-level default. Defaults to None.
            tasks: List of supported tasks. If provided, it overrides the
                class-level default. Defaults to None.
            mltypes: List of supported learning types. If provided, it overrides
                the class-level default. Defaults to None.
            model_dir: Directory of the model. If provided, it overrides the
                class-level default. Defaults to None.
        """
        # Validate inputs
        if arch is not None and not isinstance(name, str):
            raise TypeError(
                f"Expected 'arch' to be a str, but got {type(name).__name__}."
            )
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"Expected 'name' to be a str, but got {type(name).__name__}."
            )
        if tasks is not None and not isinstance(tasks, list):
            raise TypeError(
                f"Expected 'tasks' to be a list, but got {type(tasks).__name__}."
            )
        if mltypes is not None and not isinstance(mltypes, list):
            raise TypeError(
                f"Expected 'mltypes' to be a list, but got {type(mltypes).__name__}."
            )
        if model_dir is not None and not isinstance(model_dir, Path):
            raise TypeError(
                f"Expected 'model_dir' to be a Path, but got {type(model_dir).__name__}."
            )

        # Assign attributes
        # If provided, these instance variables will override the class-level defaults
        if arch is not None:
            self.arch = arch
        if name is not None:
            self.name = name
        if tasks is not None:
            # We use list() to create a copy, preventing shared state bugs
            self.tasks = list(tasks)
        if mltypes is not None:
            # We use list() to create a copy, preventing shared state bugs
            self.mltypes = list(mltypes)
        if model_dir is not None:
            self.model_dir = Path(model_dir).normalize()

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance."""
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["arch", "name", "tasks", "mltypes", "model_dir"]:
            if not hasattr(cls, attr):  # or getattr(cls, attr) is None:
                raise TypeError(
                    f"Class {cls.__name__} must define '{attr}' attribute "
                    f"(defined locally or inherited)."
                )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
