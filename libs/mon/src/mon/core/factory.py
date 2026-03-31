#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory.

This module provides factory classes.
"""

from __future__ import annotations

__all__ = [
    "ALBUMENTATIONS",
    "BACKBONES",
    "DATASETS",
    "DatasetFactory",
    "Factory",
    "MODELS",
    "ModelFactory",
    "OPTIMIZERS",
    "OptimizerFactory",
    "PREDICTORS",
    "SCHEDULERS",
    "TRAINERS",
    "UPSAMPLERS",
    "WEIGHTS",
    "WeightsFactory",
]

import inspect
from collections import UserDict
from typing import Any, Callable

from torch.optim.optimizer import Optimizer, ParamsT

from mon.core.console import log_error
from mon.core.data import Weights, WeightsEnum
from mon.core.dtype import Split, Task
from mon.core.path import Path
from mon.core.typing import PathLike, RunModeLike, TaskLike
from mon.core.utils import depascalize, is_valid_str


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Factory(UserDict[str, Any | Callable[..., Any]]):
    """Generic Factory class based on a dictionary.

    Maintain a mapping of registered classes and their names.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        mapping: dict | None = None,
        decamelize: bool = False,
        verbose: bool = False,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name for the factory.
            mapping (dict, optional): Initial dictionary of registered classes.
                Defaults to None.
            decamelize (bool, optional): If True, normalize class names to
                snake_case. Defaults to False.
            verbose (bool), optional: Verbosity mode. Defaults to False.

        Raises:
            ValueError: If ``name`` is empty.
        """
        # Validate inputs
        if not name:
            raise ValueError(
                f"Expected 'name' to be a non-empty string, but got '{name}'."
            )

        # Assign attributes
        self.verbose = verbose
        self.name = name
        self.decamelize = decamelize

        # Continue the initialization chain
        super().__init__(mapping or {})

    # --- Representation ---
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"items={list(self.items())})"
        )

    # --- Registering ---
    def register(
        self,
        name: str = "",
        module: Any = None,
        metaclass: Any = None,
        replace: bool = False,
    ) -> Callable:
        """Register a class or function.

        Args:
            name (str, optional): Name to register the ``module``. Defaults to "".
            module (Any, optional): Class or function to register. Defaults to None.
            metaclass (Any, optional): Metaclass to get metadata from.
                Defaults to None.
            replace (bool, optional): If True, overwrite an existing entry.
                Defaults to False.

        Returns:
            Callable: Decorator function if ``module`` is None, else None.
        """
        def _register(cls):
            self._register(name, cls, metaclass, replace)
            return cls

        return _register(module) if module is not None else _register

    def _register(
        self,
        name: str,
        module: Any,
        metaclass: Any = None,
        replace: bool = False,
    ):
        """Register a class or function.

        Args:
            name (str): Name to register the ``module``.
            module (Any): Class or function to register.
            metaclass (Any, optional): Metaclass to get metadata from.
                Defaults to None.
            replace (bool, optional): If True, overwrite an existing entry.
                Defaults to False.

        Raises:
            TypeError: If ``module`` is not a class or function.
            KeyError: If ``replace`` is False and the key is already registered.
        """
        # Validate inputs
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                f"Expected 'module' to be a class or function, "
                f"but got {type(module).__name__}."
            )

        # Determine the registration key.
        # Priority: explicit name > class attributes > class name.
        key = name or self._get_attr(module, metaclass, "name") or module.__name__

        # Normalize key
        if self.decamelize:
            key = depascalize(key)

        # Register the module
        if not replace and key in self:
            if self.verbose:
                log_error(
                    f"'{key}' has been already registered in the '{self.name}' "
                    f"factory. Skipping registration."
                )
            return

        self[key] = module

        # Store metadata alongside the module
        self._set_attr(module, attr="name", value=key)

    # --- Creation ---
    def build(self, name: str, *args, **kwargs) -> Any:
        """Instantiate a registered class by name.

        Args:
            name (str): Name of the registered class to instantiate.
            *args: Positional arguments to forward to the class constructor.
            **kwargs: Arguments to forward to the class constructor.

        Returns:
            Any: Instance of the requested class.

        Raises:
            ValueError: If the requested ``name`` is not found.
        """
        # Validate inputs
        if not name:
            raise ValueError(
                f"Cannot build from an empty name in the '{self.name}' factory."
            )

        # Normalize name
        key = depascalize(name) if self.decamelize else name
        if key not in self:
            # Fallback for cases where the raw name might match.
            key = name if name in self else None

        # Create the instance
        if key is None:
            raise ValueError(
                f"'{name}' is not a registered name in the '{self.name}' "
                f"factory. Available names: {list(self.keys())}."
            )
        return self._create_instance(self[key], name, *args, **kwargs)

    def _create_instance(self, cls: type, name: str, *args, **kwargs) -> Any:
        """Create an instance of the given class.

        Create an instance of the given class and set its ``name`` attribute.

        Args:
            cls (type): Class to instantiate.
            name (str): Name to set on the instance.
            *args: Positional arguments to forward to the class constructor.
            **kwargs: Arguments to forward to the class constructor.

        Returns:
            Any: Instance of the class with the ``name`` attribute set.
        """
        # Create the instance
        instance = cls(*args, **kwargs)

        # Set the name attribute
        self._set_attr(instance, attr="name", value=name)
        return instance

    # --- Mutation ---
    def sort(self, reverse: bool = False):
        """Sort the registry entries alphabetically by key.

        Args:
            reverse (bool, optional): If True, sort in descending order.
                Defaults to False.
        """
        # Sort the dictionary by key
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)

        # Rebuild the dictionary in sorted order
        self.clear()
        self.update(sorted_items)

    # --- Utils ---
    @staticmethod
    def _get_attr(module: Any, metaclass: Any, attr: str) -> Any:
        """Attempt to get an attribute from a module or metaclass.

        Args:
            module (Any): Module to get the attribute from.
            metaclass (Any): Metaclass to get the attribute from.
            attr (str): Name of the attribute to get.

        Returns:
            Any: The attribute value if found, else None.
        """
        try:
            value = getattr(module, f"{attr}")
        except AttributeError:
            value = None

        try:
            return value or getattr(metaclass, f"{attr}")
        except AttributeError:
            return None

    @staticmethod
    def _set_attr(module: Any, attr: str, value: Any):
        """Attempt to set an attribute on the module.

        Args:
            module (Any): Module to set the attribute on.
            attr (str): Name of the attribute to set.
            value (Any): Value to set the attribute to.
        """
        if inspect.isclass(module):
            try:
                if getattr(module, f"{attr}") != value:
                    setattr(module, f"{attr}", value)
            except (AttributeError, TypeError):
                # Silently fail if the attribute is not settable (e.g., on built-ins).
                pass

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class DatasetFactory(Factory):
    """Dataset Factory that organizes datasets by task.

    Extend ``Factory`` to support dataset discovery and retrieval based on tasks.
    """

    # --- Discovery ---
    def search(
        self,
        task: TaskLike | None = None,
        mode: RunModeLike | None = None
    ) -> list[str]:
        """Find all available dataset names matching a task and mode.

        Args:
            task (TaskType, optional): Task name to filter datasets.
                Defaults to None.
            mode (RunModeType, optional): Run mode to filter datasets.
                Defaults to None.

        Returns:
            list[str]: Sorted list of dataset names.
        """
        # Validate inputs
        if not any([task, mode]):
            if self.verbose:
                log_error(
                    f"Expected at least one of 'task' or 'mode', "
                    f"but got: {task}, {mode}.",
                )
            return []

        return self.filter(task=task, mode=mode)

    # --- Retrieval ---
    def filter(
        self,
        task: TaskLike | None = None,
        mode: RunModeLike | None = None
    ) -> list[str]:
        """Filter and return all available dataset names matching a task and mode.

        Args:
            task (TaskType, optional): Task name to filter datasets. If None,
                return all datasets. Defaults to None.
            mode (RunModeType, optional): Run mode to filter datasets. If None,
                return all datasets. Defaults to None.

        Returns:
            list[str]: Sorted list of dataset names.
        """
        # Global discovery
        datasets = [name for name, meta in self.items()]

        # Filter by Task
        if task in Task:
            task = Task(task)
            datasets = [d for d in datasets if task in self[d].tasks]

        # Map execution mode to data split requirements
        mode = Split(mode) if mode else None
        if mode == "train":
            required_split = Split.TRAIN
        elif mode == "val":
            required_split = Split.VAL
        else:
            required_split = Split.TEST
        datasets = [d for d in datasets if required_split in self[d].splits]

        return sorted(datasets)


class ModelFactory(Factory):
    """Model Factory that organizes models by task, architecture, and mode.

    Maintain a nested structure: ``arch -> {model_name: model_class}`` and
    provide methods for registering and building models within this structure.
    Also, include discovery and retrieval functionality for models
    and architectures.
    """

    # --- Properties ---
    @property
    def archs(self) -> list[str]:
        """Return a list of all registered architecture names."""
        return list(self.keys())

    @property
    def models(self) -> list[str]:
        """Return a flattened list of all registered model names."""
        return list(self.models_flat.keys())

    @property
    def models_flat(self) -> dict[str, dict[str, dict]]:
        """Return a flattened dictionary of all models (i.e., {'model_name': metadata}).
        """
        flatten_dict = {}
        for arch, models in self.items():
            if not isinstance(models, dict):
                continue
            for name, meta in models.items():
                flatten_dict[name] = meta

        return flatten_dict

    # --- Registering ---
    def register(
        self,
        name: str = "",
        arch: str = "",
        module: Any = None,
        metaclass: Any = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Register a model class.

        Args:
            name (str, optional): Model name. Defaults to "".
            arch (str, optional): Architecture name. Defaults to "".
            module (Any): Class or function to register. Defaults to None.
            metaclass (Any): Metaclass to get metadata from. Defaults to None.
            replace (bool): If True, overwrite an existing entry. Defaults to False.

        Returns:
            Callable: Decorator function if ``module`` is None, else None.
        """

        def _register(cls: type) -> type:
            self._register_module(name, arch, cls, metaclass, replace)
            return cls

        return _register(module) if module is not None else _register

    def _register_module(
        self,
        name: str,
        arch: str,
        module: Any,
        metaclass: Any = None,
        replace: bool = False
    ):
        """Register a class or function.

        Args:
            name (str): Model name.
            arch (str): Architecture name.
            module (Any): Class or function to register.
            metaclass (Any): Metaclass to get metadata from. Defaults to None.
            replace (bool): If True, overwrite an existing entry. Defaults to False.

        Raises:
            TypeError: If ``module`` is not a class or function.
            KeyError: If ``replace`` is False and the key is already registered.
        """
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                f"Expected 'module' to be a class or function, "
                f"but got {type(module).__name__}."
            )

        # Determine the registration key.
        # Priority: explicit name > class attributes > class name.
        arch = arch or self._get_attr(module, metaclass, "arch") or module.__name__
        model = name or self._get_attr(module, metaclass, "name") or module.__name__

        # Normalize key
        if self.decamelize:
            arch = depascalize(arch)
            model = depascalize(model)

        # Register the module
        self.setdefault(arch, {})

        if not replace and model in self[arch]:
            if self.verbose:
                log_error(
                    f"'{model}' is already registered in the '{self.name}' "
                    f"factory. Skipping registration."
                )
            return

        # Store metadata alongside the module
        self[arch][model] = {
            "arch": arch,
            "name": model,
            "tasks": self._get_attr(module, metaclass, "tasks"),
            # "mltypes": self._get_attr(module, metaclass, "mltypes"),
            "model_dir": self._get_attr(module, metaclass, "model_dir"),
            "module": module,
        }

        # Store metadata alongside the module
        self._set_attr(module, "arch", arch)
        self._set_attr(module, "name", model)

    # --- Creation ---
    def build(self, name: str, *args, **kwargs) -> Any:
        """Instantiate a registered model by name.

        Args:
            name (str): Name of the registered model to instantiate.
            *args: Positional arguments to forward to the model constructor.
            **kwargs: Arguments to forward to the model constructor.

        Returns:
            Any: Instance of the requested model.

        Raises:
            ValueError: If the requested model ``name`` is not found.
        """
        # Validate inputs
        if not name:
            raise ValueError(
                f"Cannot build from an empty name in the '{self.name}' factory."
            )

        flatten = self.models_flat

        # Normalize name
        key = depascalize(name) if self.decamelize else name
        if key not in flatten:
            # Fallback for cases where the raw name might match.
            key = name if name in flatten else None

        # Create the instance
        if key is None:
            raise ValueError(
                f"'{name}' is not a registered name in the '{self.name}' "
                f"factory. Available names: {list(flatten.keys())}."
            )
        return self._create_instance(flatten[key]["module"], key, *args, **kwargs)

    # --- Discovery ---
    def search(self, arch: str = "", task: TaskLike | None = None) -> list[str]:
        """Find all available model names matching an architecture, task, or
        run mode.

        Args:
            arch (str, optional): Architecture name to filter datasets.
                Defaults to "".
            task (TaskType, optional): Task name to filter datasets.
                Defaults to None.

        Returns:
            list[str]: Sorted list of model names.
        """
        if not arch and not task:
            if self.verbose:
                log_error(
                    f"Expected at least one of 'arch' or 'task',"
                    f"but got: {arch}, {task}.",
                )
            return []

        return self.filter(arch=arch, task=task)

    def search_archs(self, task: TaskLike | None = None) -> list[str]:
        """Find all available architectures matching a task and mode.

        Args:
            task (TaskType, optional): Task name to filter datasets.
                Defaults to None.

        Returns:
            list[str]: Sorted list of architecture names.
        """
        # Get all models names matching the task and mode
        models = self.search(task=task)

        # Resolve architecture names from the registry
        flatten = self.models_flat
        archs = set()  # Use set to automatically handle duplicates

        for m in models:
            # Check if the model exists in the registry and has an 'arch'
            # attribute
            entry = flatten[m]
            arch = entry.get("arch")
            arch = str(arch).strip() if arch else None
            if arch and arch.lower() != "none":
                archs.add(arch)

        return sorted(list(archs))

    def find_module(self, name: str) -> Any | None:
        """Find the module for a model by name and optionally by task, mode,
        and architecture.

        Args:
            name (str): Model name to find.

        Returns:
            Any | None: The module if found, else None.
        """
        # Normalize key
        key = depascalize(name) if self.decamelize else name

        # Find the module for the given name
        flatten = self.models_flat

        if key in flatten:
            return flatten[key].get("module")

        return None

    # --- Validation ---
    def has_model(self, name: str) -> bool:
        """Check if a model has been registered with the given name.

        Args:
            name (str): Model name to check.
        """
        # Normalize key
        key = depascalize(name) if self.decamelize else name
        return key in self.models_flat

    # --- Retrieval ---
    def get_model(self, name: str) -> dict[str, Any] | None:
        """Return the metadata for a registered model by name."""
        # Normalize key
        key = depascalize(name) if self.decamelize else name

        # Look up the model metadata in the registry
        flatten = self.models_flat
        if key in flatten:
            return flatten[key]

        return None

    def get_model_dir(self, name: str) -> Path | None:
        """Return the absolute path to the model definition directory."""
        # Check if the model is registered
        model_entry = self.get_model(name=name)
        if not model_entry:
            return None

        # Get the model directory
        model_dir = model_entry.get("model_dir")
        if not model_dir:
            return None

        # Check if the directory exists
        model_dir = Path(model_dir).normalize()
        if model_dir.is_dir():
            return Path(model_dir)
        else:
            return None

    def filter(self, arch: str = "", task: TaskLike | None = None) -> list[str]:
        """Filter and return all available model names matching an architecture,
        or task.

        Args:
            arch (str, optional): Architecture name to filter datasets.
                Defaults to "".
            task (TaskType, optional): Task name to filter datasets.
                Defaults to None.

        Returns:
            list[str]: Sorted list of model names.
        """
        # Global discovery
        flatten = self.models_flat
        models = list(self.models_flat.keys())

        # Filter by Architecture
        if arch:
            models = [m for m in models if arch == flatten[m]["arch"]]

        # Filter by Task
        if task in Task:
            task = Task(task)
            models = [m for m in models if task in flatten[m]["tasks"]]

        return sorted(models)


class WeightsFactory(Factory):
    """Weights Factory that organizes pretrained weights."""

    # --- Properties ---
    @property
    def weights_objs(self) -> list[Weights]:
        """Return a list of all registered weights objects."""
        weights_objs = []
        for w_enum in self.values():
            weights_objs.extend(w_enum.values())
        return weights_objs

    # --- Discovery ---
    def find(self, weights_path: PathLike | None) -> WeightsEnum | None:
        """Find the ``WeightsEnum`` object for a given path.

        Args:
            weights_path (PathLike): Path to look for the weights enum.

        Returns:
            WeightsEnum: ``WeightsEnum`` object if found, None otherwise.
        """
        if weights_path:
            weights_path = Path(weights_path).normalize()
        else:
            weights_path = None

        # Global Discovery
        if weights_path:
            for w_enum in self.values():
                for w in w_enum.values():
                    if weights_path == w.path:
                        return w_enum(w)

        # Return None if no match is found
        return None

    def find_weights_obj(self, weights_path: PathLike | None) -> Weights | None:
        """Find the ``Weights`` object for a given path.

        Args:
            weights_path (PathLike): Path to look for the weights object.

        Returns:
            Weights: ``Weights`` object if found, None otherwise.
        """
        if is_valid_str(weights_path):
            weights_path = Path(weights_path).normalize()
        else:
            weights_path = None

        # Global Discovery
        if weights_path:
            for w in self.weights_objs:
                if w.path == weights_path:
                    return w

        # Return None if no match is found
        return None

    # --- Validation ---
    def has(self, weights_path: PathLike) -> bool:
        """Check if there is a weights enum registered for a given path.

        Args:
            weights_path (PathLike): Path to look for the weights enum.
        """
        return self.find(weights_path) is not None


class OptimizerFactory(Factory):
    """Optimizer Factory that organizes optimizers."""

    # --- Creation ---
    def build(self, name: str, params: ParamsT, *args, **kwargs) -> Optimizer:
        """Instantiate a registered class by name.

        Args:
            name (str): Name of the registered class to instantiate.
            params (ParamsT): Parameters to pass to the class constructor.
            *args: Positional arguments to forward to the class constructor.
            **kwargs: Arguments to forward to the class constructor.

        Returns:
            Any: Instance of the requested class.

        Raises:
            ValueError: If the requested ``name`` is not found.
        """
        # Validate inputs
        if not name:
            raise ValueError(
                f"Cannot build from an empty name in the '{self.name}' factory."
            )

        # Normalize name
        key = depascalize(name) if self.decamelize else name
        if key not in self:
            # Fallback for cases where the raw name might match.
            key = name if name in self else None

        # Create the instance
        if key is None:
            raise ValueError(
                f"'{name}' is not a registered name in the '{self.name}' "
                f"factory. Available names: {list(self.keys())}."
            )
        return self[key](params=params, *args, **kwargs)

# endregion


# ==============================================================================
# region CONSTANTS
# ==============================================================================

ALBUMENTATIONS: Factory = Factory(name="Albumentations", decamelize=False)
DATASETS: DatasetFactory = DatasetFactory(name="Datasets", decamelize=False)

BACKBONES: ModelFactory = ModelFactory(name="Backbones", decamelize=False)
UPSAMPLERS: Factory = Factory(name="Upsamplers", decamelize=False)
MODELS: ModelFactory = ModelFactory(name="Models", decamelize=False)
WEIGHTS: WeightsFactory = WeightsFactory(name="Weights", decamelize=False)
OPTIMIZERS: OptimizerFactory = OptimizerFactory(name="Optimizers", decamelize=False)
SCHEDULERS: Factory = Factory(name="Schedulers", decamelize=False)

TRAINERS: Factory = Factory(name="Trainers", decamelize=False)
PREDICTORS: Factory = Factory(name="Predictors", decamelize=False)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
