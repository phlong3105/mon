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
        arch (str): The model's architecture. `Must be defined in subclasses or
            set during initialization.`
        name (str): The model's name. `Must be defined in subclasses or set
            during initialization.`
        tasks (list[Task]): A list of tasks that the model can perform. `Must be
            defined in subclasses or set during initialization.`
        mltypes (list[MLType]): A list of learning types that the model can
            perform. `Must be defined in subclasses or set during initialization.`
        model_dir (Path): The directory where the model is stored. `Must be
            defined in subclasses or set during initialization.`
    """

    arch: str
    name: str
    tasks: list[Task]
    mltypes: list[MLType]
    model_dir: Path

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        arch: str | None = None,
        name: str | None = None,
        tasks: list[Task] | None = None,
        mltypes: list[MLType] | None = None,
        model_dir: Path | str | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            arch (str, optional): The model's architecture. If provided, it
                overrides the class-level default. Defaults to None.
            name (str, optional): The model's name. If provided, it overrides
                the class-level default. Defaults to None.
            tasks (list[Task], optional): A list of tasks that the model can
                perform. If provided, it overrides the class-level default.
                Defaults to None.
            mltypes (list[MLType], optional): A list of learning types that the
                model can perform. If provided, it overrides the class-level
                default. Defaults to None.
            model_dir (Path | str, optional): The directory where the model is
                stored. If provided, it overrides the class-level default.
                Defaults to None.
        """
        # Assign attributes
        if isinstance(arch, str):
            self.arch = arch
        if isinstance(name, str):
            self.name = name
        if isinstance(tasks, list) and all(isinstance(t, Task) for t in tasks):
            # We use list() to create a copy, preventing shared state bugs
            self.tasks = list(tasks)
        if isinstance(mltypes, list) and all(isinstance(m, MLType) for m in mltypes):
            # We use list() to create a copy, preventing shared state bugs
            self.mltypes = list(mltypes)
        if isinstance(model_dir, (Path, str)):
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
