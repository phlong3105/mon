#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes and mixins for data containers."""

from __future__ import annotations

__all__ = [
    "Dataset",
    "Modalities",
    "Modality",
    "RegistrableMixin",
]

import abc
from typing import Any, Dict, NamedTuple, TypeAlias

from torch.utils.data import dataset

from mon.core import log, Path, Task
from mon.core.dtypes import ClassList


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---

class Modality(NamedTuple):
    """Data structure representing a modality in a dataset.

    Attributes:
        name: Name of the directory that contains the modality data.
        type: Albumentations target type for augmentations.
        module: Tensor class that performs I/O operations.
        train: If True, this modality is included in the train/val set.
        test: If True, this modality is included in the test set.
        primary: If True, this is the primary modality.
    """

    name   : str
    type   : str | None = None
    module : Any        = None
    train  : bool       = True
    test   : bool       = False
    primary: bool       = False


Modalities: TypeAlias = Dict[str, Modality]


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Dataset(dataset.Dataset, abc.ABC):
    """Abstract class for all datasets.

    Subclass this abstract class to create specific dataset implementations.

    Attributes:
        datapoints: Dictionary containing lists of datapoints for each modality.
            `Must be initialized during subclass initialization.`
        classlist: Dataset object classes. `Should be defined in subclasses or
            set during initialization.`
        verbose: If True, enable verbose output.
    """

    classlist: ClassList | None = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        datapoints: dict[str, list[Any]] | None = None,
        classlist : Path | ClassList     | None = None,
        verbose   : bool                        = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            datapoints: Dictionary containing lists of datapoints for each
                modality. Defaults to None.
            classlist: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            TypeError: If ``datapoints`` is not a dict or None.
            TypeError: If ``verbose`` is not a bool.
        """
        super().__init__(*args, **kwargs)

        # Validate inputs
        if datapoints is not None and not isinstance(datapoints, dict):
            raise TypeError(
                f"Expected 'datapoints' to be a dict or None, "
                f"but got {type(datapoints).__name__}."
            )
        if not isinstance(verbose, bool):
            raise TypeError(
                f"Expected 'verbose' to be a bool, "
                f"but got {type(verbose).__name__}."
            )

        # Assign attributes
        self.verbose    = verbose
        self.datapoints = datapoints or {}
        self.set_classlist(classlist)

    @abc.abstractmethod
    def __del__(self):
        """Finalize the object.

        Close the dataset loading mechanism and release resources.
        """
        pass

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        lines  = [f"Dataset {self.__class__.__name__}"]
        lines += [f"Number of datapoints: {len(self)}"]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the container."""
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index: Index to access.
        """
        pass

    def __iter__(self):
        """Return an iterator for the container."""
        for i in range(len(self)):
            yield self[i]

    # --- Properties ---
    def set_classlist(self, value: Path | ClassList | None):
        """Set the dataset's class definitions.

        Args:
            value: Either a .yaml file containing the classes definitions,
                or a ClassList instance. Defaults to None.

        Raises:
            TypeError: If ``value`` is not a valid type.
        """
        if value is None:
            self.classlist = None
        elif isinstance(value, (Path, ClassList)):
            self.classlist = ClassList(value)
            if self.verbose:
                log(f"'classlist' set with {len(self.classlist)} classes.")
        else:
            raise TypeError(
                f"Expected 'value' to be a Path, ClassList, or None, "
                f"but got {type(value).__name__}."
            )

    @property
    def disable_pbar(self) -> bool:
        """Check if progress bars are disabled."""
        return not self.verbose

    # --- Access ---
    @abc.abstractmethod
    def _get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.
        """
        pass

    def _get_underlying_data(self, index: int) -> dict[str, Any]:
        """Get the underlying data of a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Raises:
            TypeError: If ``index`` is not an int.
            IndexError: If ``index`` is out of range.
        """
        if not isinstance(index, int):
            raise TypeError(f"Expected 'index' to be an int, but got {type(index).__name__}.")
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        datapoint = self._get_datapoint(index=index)
        # Optimized 'data' extraction using dictionary comprehension
        return {
            k: getattr(v, "data", v) if v is not None else None
            for k, v in datapoint.items()
        }


# --- Mixins ---

class RegistrableMixin(abc.ABC):
    """A mixin class that adds metadata attribute to datasets for factory
    registration purposes.

    Attributes:
        name: Name of the data container. `Must be defined in subclasses or set
            during initialization.`
        tasks: List of supported tasks. `Must be defined in subclasses or set
            during initialization.`
    """

    name : str | None = None
    tasks: list[Task] = []

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name : str        | None = None,
        tasks: list[Task] | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the data container. If provided, it overrides the
                class-level default. Defaults to None.
            tasks: List of supported tasks. If provided, it overrides the
                class-level default. Defaults to None.
        """
        # Validate inputs
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Expected 'name' to be a str, but got {type(name).__name__}.")
        if tasks is not None and not isinstance(tasks, list):
            raise TypeError(f"Expected 'tasks' to be a list, but got {type(tasks).__name__}.")

        # Assign attributes
        # If provided, these instance variables will override the class-level defaults
        if name is not None:
            self.name = name
        if tasks is not None:
            # We use list() to create a copy, preventing shared state bugs
            self.tasks = list(tasks)

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance."""
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["name", "tasks"]:
            if not hasattr(cls, attr) or getattr(cls, attr) is None:
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
