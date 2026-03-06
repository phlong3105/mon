#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model Mixins.

This module defines mixins for models to extend their functionality.
"""

from __future__ import annotations

__all__ = [
    "ModelRegisterMixin",
]

from abc import ABC

from mon.core import is_list_of, is_valid_str, Path, PathLike, Task


# ==============================================================================
# region CREATION
# ==============================================================================

class ModelRegisterMixin(ABC):
    """A mixin class for models that can be registered in a factory.

    Attributes:
        name (str, optional): Name of the dataset. Defaults to "" and should be
            overridden in subclasses or set during initialization.
        tasks (list[Task]): List of supported tasks. Defaults to an empty list
            and should be overridden in subclasses or set during initialization.
    """

    arch: str = ""
    name: str = ""
    tasks: list[Task] = []
    model_dir: Path = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        arch: str = "",
        name: str = "",
        tasks: list[Task] | None = None,
        model_dir: PathLike | None = None,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            arch (str, optional): The model's architecture. If provided, it
                overrides the class-level default. Defaults to None.
            name (str, optional): Name of the data container. If provided, it
                overrides the class-level default. Defaults to "".
            tasks (list[Task], optional): List of supported tasks. If provided,
                it overrides the class-level default. Defaults to None.
            model_dir (PathLike, optional): Directory where the model is
                defined. If provided, it overrides the class-level default.
                Defaults to None.
            *args: Positional arguments to forward to the superclass constructor.
            **kwargs: Keyword arguments to forward to the superclass constructor.
        """
        # Assign attributes
        if is_valid_str(arch):
            self.arch = arch
        if is_valid_str(name):
            self.name = name
        if is_list_of(tasks, Task):
            # We use list() to create a copy, preventing shared state bugs
            self.tasks = list(tasks)
        if is_valid_str(model_dir):
            model_dir = Path(model_dir).normalize()
            if model_dir.is_dir():
                self.model_dir = model_dir

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Raises:
            AttributeError: If the subclass does not define ``name`` or ``tasks``
                attributes (either locally or inherited).
        """
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["name", "tasks", "model_dir"]:
            if not getattr(cls, attr):
                raise AttributeError(
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
