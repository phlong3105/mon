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

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, OrderedDict, override, TypeAlias

from torch.utils.data.dataset import Dataset as Dataset_

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
        name (str): Name of the modality.
        module (Any): The class associated with the modality.
        type (str, optional): Type of the modality (e.g., "image", "text") for
            albumentations augmentations.
        test (bool): Indicates if the modality is available for testing
            (i.e., ground-truth).
    """

    name: str
    module: Any
    type: str | None = None
    test: bool = True


Modalities: TypeAlias = OrderedDict[str, Modality]


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Dataset(Dataset_, ABC):
    """Abstract class for all datasets.

    Subclass this abstract class to create specific dataset implementations.

    Attributes:
        classlist (ClassList, optional): Class definitions for the dataset.
            `Should be defined in subclasses or set during initialization.`
    """

    classlist: ClassList | None = None

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        classlist: ClassList | Path | str | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.
            verbose (bool): Verbosity mode. Defaults to True.

        Raises:
            TypeError: If ``datapoints`` is not a dict or None.
            TypeError: If ``verbose`` is not a bool.
        """
        super().__init__(*args, **kwargs)

        # Assign attributes
        self.verbose = verbose
        self.datapoints: OrderedDict[str, list[Any]] = OrderedDict()
        self.set_classlist(classlist)

    @abstractmethod
    def __del__(self):
        """Finalize the object.

        Close the dataset loading mechanism and release resources.
        """
        pass

    # --- Representation ---
    @override
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        lines = [f"Dataset {self.__class__.__name__}"]
        lines += [f"Number of datapoints: {len(self)}"]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the container."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        pass

    def __iter__(self):
        """Return an iterator for the container."""
        for i in range(len(self)):
            yield self[i]

    # --- Properties ---
    def set_classlist(self, classlist: ClassList | Path | str | None):
        """Set the dataset's class definitions.

        Args:
            classlist (ClassList | Path | str, optional): Class definitions for
                the dataset. Can be a ``ClassList`` instance or a path to a
                .yaml file. Defaults to None.

        Raises:
            TypeError: If ``value`` is not a valid type.
        """
        if classlist is None:
            self.classlist = None
        elif isinstance(classlist, (ClassList, Path, str)):
            self.classlist = ClassList(classlist)
            if self.verbose:
                log(f"'classlist' set with {len(self.classlist)} classes.")
        else:
            raise TypeError(
                f"Expected 'value' to be a ClassList or Path, but got "
                f"{type(classlist).__name__}."
            )

    @property
    def disable_pbar(self) -> bool:
        """Check if progress bars are disabled."""
        return not self.verbose

    # --- Access ---
    @abstractmethod
    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        pass

    def get_underlying_data(self, index: int) -> dict[str, Any]:
        """Get the underlying data of a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        datapoint = self.get_datapoint(index=index)
        for k, v in datapoint.items():
            if v is not None and hasattr(v, "data"):
                datapoint[k] = v.data

        return datapoint


# --- Mixins ---

class RegistrableMixin(ABC):
    """A mixin class that adds metadata attribute to datasets for factory
    registration purposes.

    Attributes:
        name (str): Name of the data container. `Must be defined in subclasses
            or set during initialization.`
        tasks (list[Task]): List of supported tasks. `Must be defined in
            subclasses or set during initialization.`
    """

    name: str
    tasks: list[Task]

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str | None = None,
        tasks: list[Task] | None = None,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the data container. If provided, it overrides
                the class-level default. Defaults to None.
            tasks (list[Task]): List of supported tasks. If provided, it overrides
                the class-level default. Defaults to None.
        """
        # Assign attributes
        if isinstance(name, str):
            self.name = name
        if isinstance(tasks, list) and all(isinstance(t, Task) for t in tasks):
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
