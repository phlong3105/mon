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
from typing import Any

from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torch import nn, Tensor

from mon.core import is_list_of, is_valid_str, Path, PathLike, SizeLike, Task


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Model(nn.Module, ABC):
    """A base class for all deep learning models.

    Attributes:
        in_keys (set): A set of strings representing the input keys that the
            model expects. Subclasses should override this to define their
            input specifications.
        out_keys (set): A set of strings representing the output keys that the
            model produces. Subclasses should override this to define their
            output specifications.
    """

    in_keys: set = {}
    out_keys: set = {}

    # --- Lifecycle & Initialization ---
    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Raises:
            AttributeError: If ``name`` or ``splits`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["in_keys", "out_keys"]:
            if not getattr(cls, attr):
                raise AttributeError(
                    f"Class {cls.__name__} must define '{attr}' attribute "
                    f"(defined locally or inherited)."
                )

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  [Contract]\n"
            f"    in_keys : {sorted(list(self.in_keys))}\n"
            f"    out_keys: {sorted(list(self.out_keys))}\n"
            f")"
        )

    # --- Callable & Context Manager ---
    def forward(
        self,
        data: TensorDict | None = None,
        save_debug: bool = False,
        *args, **kwargs
    ) -> TensorDict:
        """Forward the input through the model.

        Args:
            data (TensorDict, optional): Input data dictionary. Defaults to None.
            save_debug (bool, optional): If True, return intermediate results
                for debugging. Defaults to False.
            **kwargs: Direct keyword arguments to pass to the forward step.
                Useful for simple inference, but should be used with caution as
                it bypasses the input validation. It's recommended to use the
                ``data`` dictionary for structured inputs.

        Returns:
            TensorDict: Output dictionary.
        """
        # 1. Split kwargs into Data (for TensorDict) and Flags (for logic)
        # data_kwargs go into the TensorDict, flags stay for the method call
        data_kwargs = {k: v for k, v in kwargs.items() if k in self.in_keys}
        flags = {k: v for k, v in kwargs.items() if k not in data_kwargs}

        # 2. Convert data input to TensorDict
        data_kwargs = {
            k: v if isinstance(v, Tensor) else NonTensorData(v)
            for k, v in data_kwargs.items()
        }

        if data is None:
            data = TensorDict(data_kwargs, batch_size=[])
        elif isinstance(data, TensorDict):
            data.update(data_kwargs)
        else:
            raise TypeError(
                f"Expected 'data' to be TensorDict, but got {type(data).__name__}."
            )

        # 3. Contract enforcement (Input)
        missing_inputs = self.in_keys - data.keys()
        if missing_inputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required inputs: {missing_inputs}. "
                f"Provided keys: {list(data.keys())}"
            )

        # 4. Execution
        outputs = self.forward_step(data=data, **flags)

        # 5. Ensure outputs is a TensorDict
        if not isinstance(outputs, TensorDict):
            outputs = TensorDict(outputs, batch_size=[])

        # 6. Contract enforcement (Output)
        missing_outputs = self.out_keys - outputs.keys()
        if missing_outputs:
            raise KeyError(
                f"{self.__class__.__name__} missing required outputs: {missing_outputs}. "
                f"Provided keys: {list(outputs.keys())}"
            )

        # 7. Filtering & return
        if not save_debug:
            # select() returns a new TensorDict with only the 'provides' keys
            outputs = outputs.select(*self.out_keys, strict=False)

        return outputs

    @abstractmethod
    def forward_step(self, data: TensorDict, *args, **kwargs) -> TensorDict:
        """Perform a single forward step of the model.

        This method should be implemented by all subclasses.

        Args:
            data (TensorDict): Input data dictionary.

        Returns:
            TensorDict: Output data dictionary.
        """
        pass

    # --- Benchmarks ---
    @abstractmethod
    def benchmark(self, imgsz: SizeLike, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (SizeLike): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
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
