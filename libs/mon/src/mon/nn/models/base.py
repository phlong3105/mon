#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models and Mixins.

This module defines the base class and mixins for deep learning models.
"""

from __future__ import annotations

__all__ = [
    "Model",
    "ModelRegisterMixin",
]

from abc import ABC, abstractmethod

from torch import nn, Tensor

from mon.core import is_list_of, is_valid_str, Path, PathLike, Task


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Model(nn.Module, ABC):
    """A base class for all deep learning models.

    Attributes:
        requires (dict): A dictionary specifying the required input keys and their
            expected types. Subclasses should override this to define their input
            requirements.
        provides (dict): A dictionary specifying the output keys and their expected
            types that the model will produce. Subclasses should override this to
            define their output specifications.
    """

    requires: dict = {}
    provides: dict = {}

    # --- Lifecycle & Initialization ---
    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Raises:
            AttributeError: If ``name`` or ``splits`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["requires", "provides"]:
            if not getattr(cls, attr):
                raise AttributeError(
                    f"Class {cls.__name__} must define '{attr}' attribute "
                    f"(defined locally or inherited)."
                )

    # --- Callable & Context Manager ---
    def forward(self, data: dict, save_debug: bool = False, *args, **kwargs) -> dict:
        """Forward the input through the model.

        Args:
            data (dict | Tensor): Input data dictionary or input tensor of
                shape (B, ...) and values ranging from 0.0 to 1.0.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.

        Returns:
            dict: Output dictionary.
        """
        # Validate inputs
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected `data` to be a dict, but got {type(data).__name__}."
            )

        missing_inputs = self.requires - data.keys()
        if missing_inputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required inputs: {missing_inputs}. "
                f"Provided keys: {list(data.keys())}"
            )

        # Execute forward pass logic
        outputs = self.forward_step(data=data)

        # Validate outputs
        if isinstance(outputs, dict):
            missing_outputs = self.provides - outputs.keys()
            if missing_outputs:
                raise KeyError(
                    f"{self.__class__.__name__} missing required outputs: {missing_outputs}. "
                    f"Provided keys: {list(outputs.keys())}"
                )
        else:
            raise TypeError(
                f"Expected `outputs` to be a dict, but got {type(outputs).__name__}."
            )

        # Return outputs
        if not save_debug:
            # Filter outputs to only include keys defined in `provides`
            outputs = {k: v for k, v in outputs.items() if k in self.provides}

        return outputs

    @abstractmethod
    def forward_step(self, data: dict, *args, **kwargs) -> dict:
        """Perform a single forward step of the model.

        This method should be implemented by all subclasses.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Output data dictionary.
        """
        pass

# endregion


# ==============================================================================
# region MIXINS
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
