#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class registration and instantiation utilities.

This module provides Factory and ModelFactory classes for registering and
instantiating classes.
"""

from __future__ import annotations

__all__ = [
    "DatasetFactory",
    "Factory",
    "ModelFactory",
    "WeightsFactory",
    # Constants
    "ALBUMENTATIONS",
    "BACKBONES",
    "DATASETS",
    "MODELS",
    "WEIGHTS",
]

import inspect
from typing import Any, Callable

from mon.core.console import log_error
from mon.core.types import Weights, WeightsEnum
from mon.core.enum import MLType, Split, Task
from mon.core.pathlib import Path
from mon.core.utils import depascalize


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class Factory(dict):
    """Dictionary-backed factory for class registration and construction.

    Allow classes to be registered with a specific name and later instantiated
    using that name. Support name normalization and dynamic registration via
    decorators.

    Attributes:
        _name (str): Factory name.
        _decamelize (bool): If True, normalize class names to snake_case.
            Defaults to False.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name      : str,
        mapping   : dict | None = None,
        decamelize: bool        = False,
        verbose   : bool        = False,
    ):
        """Initialize a new instance.

        Args:
            name: Name for the factory.
            mapping: Optional initial dictionary of registered classes.
                Defaults to None.
            decamelize: If True, normalize class names to snake_case.
                Defaults to False.
            verbose: Verbosity mode. Defaults to False.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError(f"Expected 'name' to be a non-empty string, but got '{name}'.")

        self.verbose     = verbose
        self._name       = name
        self._decamelize = decamelize
        super().__init__(mapping or {})

    # --- Representation ---
    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"{self.__class__.__name__}(name='{self._name}', items={list(self.items())})"

    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the factory's name."""
        return self._name

    # --- Registering ---
    def register(
        self,
        name     : str | None = None,
        module   : Any | None = None,
        metaclass: Any | None = None,
        replace  : bool       = False,
    ) -> Callable:
        """Register a class or return a decorator for registration.

        Args:
            name: Optional name to register the class under. Defaults to None.
            module: Class to register immediately. Defaults to None.
            metaclass: Metaclass to get metadata from. Defaults to None.
            replace: If True, overwrite any existing registration for the name.
                Defaults to False.

        Returns:
            Decorator if ``module`` is None, otherwise the registered class.
        """
        def _register(cls):
            self._register_module(
                module    = cls,
                name      = name,
                metaclass = metaclass,
                replace   = replace
            )
            return cls

        return _register(module) if module is not None else _register

    def _register_module(
        self,
        module   : Any,
        name     : str | None = None,
        metaclass: Any | None = None,
        replace  : bool       = False
    ):
        """Register a class internally.

        Args:
            module: Class or function to register.
            name: Optional key to register under. Defaults to None.
            metaclass: Metaclass to get metadata from. Defaults to None.
            replace: If True, overwrite an existing entry. Defaults to False.

        Raises:
            TypeError: If ``module`` is not a class or function.
            KeyError: If ``replace`` is False and the key is already registered.
        """
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                f"Expected 'module' to be a class or function, but got {type(module).__name__}."
            )

        # Determine the registration key. Priority: explicit name > class attributes > class name.
        key = name or getattr(module, "_name", getattr(module, "name", module.__name__))

        if self._decamelize:
            key = depascalize(key)

        if not replace and key in self:
            if self.verbose:
                log_error(f"'{key}' is already registered in the '{self._name}' factory.")
            return

        self[key] = module
        self._try_set_attr(module, attr_name="name", value=key)

    # --- Creation ---
    def build(self, name: str, *args, **kwargs) -> Any:
        """Instantiate a registered class by name.

        Args:
            name: Registered key of the class to instantiate.
            *args: Positional arguments to forward to the class constructor.
            **kwargs: Arguments to forward to the class constructor.

        Returns:
            Instance of the registered class.

        Raises:
            ValueError: If the requested ``name`` is not found or is empty.
        """
        if not name:
            raise ValueError(f"Cannot build from an empty name in the '{self._name}' factory.")

        key = depascalize(name) if self._decamelize else name
        if key not in self:
            # Fallback for cases where the raw name might match.
            key = name if name in self else None

        if key is None:
            raise ValueError(
                f"'{name}' is not a registered name in the '{self._name}' factory. "
                f"Available names: {list(self.keys())}."
            )

        return self._create_instance(self[key], name, *args, **kwargs)

    def _create_instance(self, cls: type, name: str, *args, **kwargs) -> Any:
        """Instantiate and tag the object with its registered name internally."""
        instance = cls(*args, **kwargs)
        self._try_set_attr(instance, attr_name="name", value=name)
        return instance

    # --- Mutation ---
    def sort(self, reverse: bool = False):
        """Sort the registry entries alphabetically by key.

        Args:
            reverse: If True, sort in descending order. Defaults to False.
        """
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        self.clear()
        self.update(sorted_items)

    # --- Utils ---
    @staticmethod
    def _try_get_attr(module: Any, metaclass: Any, attr_name: str) -> Any:
        """Attempt to get an attribute from a module or metaclass."""
        try:
            return (
                   getattr(module,    f"_{attr_name}", getattr(module,    f"{attr_name}", None))
                or getattr(metaclass, f"_{attr_name}", getattr(metaclass, f"{attr_name}", None))
            )
        except AttributeError:
            return None

    @staticmethod
    def _try_set_attr(module: Any, attr_name: str, value: Any):
        """Attempt to set an attribute on the module."""
        if inspect.isclass(module):
            try:
                if not hasattr(module, f"{attr_name}") or getattr(module, f"{attr_name}") != value:
                    setattr(module, f"{attr_name}", value)
                if not hasattr(module, f"_{attr_name}") or getattr(module, f"_{attr_name}") != value:
                    setattr(module, f"_{attr_name}", value)
            except (AttributeError, TypeError):
                # Silently fail if the attribute is not settable (e.g., on built-ins).
                pass


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class DatasetFactory(Factory):
    """Factory specialized for organizing datasets by task.

    Extend ``Factory`` to include discovery and retrieval methods specific to
    datasets, such as filtering by task and run mode.
    """

    # --- Discovery ---
    def search(
        self,
        task: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Find all available dataset names matching a task and mode.

        Args:
            task: Task name to filter models. Defaults to None.
            mode: Run mode to filter models. Defaults to None.

        Returns:
            Sorted list of model names.
        """
        if not any([task, mode]):
            if self.verbose:
                log_error(
                    f"Expected at least one of 'task' or 'mode', but got: {task}, {mode}."
                )
            return []

        return self.filter(task=task, mode=mode)

    # --- Retrieval ---
    def filter(
        self,
        task: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Filter and return all available dataset names matching a task and mode.

        Args:
            task: Task name to filter datasets. If None, return all datasets.
                Defaults to None.
            mode: Run mode to filter datasets. If None, return all datasets.
                Defaults to None.

        Returns:
            Sorted list of dataset names.
        """
        # Global Discovery
        datasets = [name for name, meta in self.items()]

        # Filter by Task (e.g., Segmentation, Classification)
        if task and task in Task.values():
            task_enum = Task(task)
            datasets  = [d for d in datasets if task_enum in self[d].tasks]

        # Map execution mode to data split requirements
        if mode == "train":
            required_split = Split.TRAIN
        elif mode == "val":
            required_split = Split.VAL
        else:
            required_split = Split.TEST
        datasets = [d for d in datasets if required_split in self[d].splits]

        return sorted(datasets)


class ModelFactory(Factory):
    """Factory specialized for organizing models by architecture.

    Maintain a nested structure: ``arch -> {model_name: model_class}``. Provide
    specialized methods for registering and building models within this
    structure. Also, include discovery and retrieval functionality for models
    and architectures.

    Examples:
        self = {
            "yolo": {
                "yolov11": {
                    "arch": ...,
                    "model": ...,
                    "tasks": ...,
                    "module": ...,
                    ...
                }
            }
        }
    """

    # --- Properties ---
    @property
    def archs(self) -> list[str]:
        """Return a list of all registered architecture names."""
        return list(self.keys())

    @property
    def models(self) -> list[str]:
        """Return a flattened list of all registered model names."""
        return [
            model_name
            for arch_models in self.values()
            if isinstance(arch_models, dict)
            for model_name in arch_models
        ]

    @property
    def flatten_dict(self) -> dict:
        """Return a flattened dictionary of all models."""
        flat_dict = {}
        for arch, models in self.items():
            if not isinstance(models, dict):
                continue
            for model_name, model_data in models.items():
                flat_dict[model_name] = model_data

        # print(flat_dict)
        return flat_dict

    # --- Registering ---
    def register(
        self,
        name     : str  | None = None,
        arch     : str  | None = None,
        variant  : str  | None = None,
        module   : Any  | None = None,
        metaclass: Any  | None = None,
        replace  : bool        = False,
    ) -> Callable[[type], type]:
        """Register a model class or return a decorator for registration.

        Args:
            name: Full model name. Defaults to None.
            arch: Architecture name. Defaults to None.
            variant: Model variant name. Defaults to None.
            module: Class to register immediately. Defaults to None.
            metaclass: Metaclass to get metadata from. Defaults to None.
            replace: If True, overwrite any existing registration.
                Defaults to False.

        Returns:
            Decorator or the registered class.
        """
        def _register(cls: type) -> type:
            self._register_module(
                module    = cls,
                name      = name,
                arch      = arch,
                variant   = variant,
                metaclass = metaclass,
                replace   = replace,
            )
            return cls

        return _register(module) if module is not None else _register

    def _register_module(
        self,
        module   : Any,
        name     : str  | None = None,
        arch     : str  | None = None,
        variant  : str  | None = None,
        metaclass: Any  | None = None,
        replace  : bool        = False
    ):
        """Register a model class internally.

        Args:
            module: Class or function to register.
            name: Full model name. Defaults to None.
            arch: Architecture name. Defaults to None.
            variant: Model variant name. Defaults to None.
            metaclass: Metaclass to get metadata from. Defaults to None.
            replace: If True, overwrite an existing entry. Defaults to False.

        Raises:
            TypeError: If ``module`` is not a class or function.
        """
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                f"Expected 'module' to be a class or function, but got {type(module).__name__}."
            )

        arch_name  = arch or self._try_get_attr(module, metaclass, "arch") or module.__name__
        model_name = name or self._try_get_attr(module, metaclass, "name") or (f"{arch_name}_{variant}" if variant else arch_name)

        if self._decamelize:
            arch_name  = depascalize(arch_name)
            model_name = depascalize(model_name)

        self.setdefault(arch_name, {})

        if not replace and model_name in self[arch_name]:
            if self.verbose:
                log_error(f"'{model_name}' is already registered in the '{self._name}' factory.")
            return

        # Store metadata alongside the module
        self[arch_name][model_name] = {
            "arch"     : arch_name,
            "name"     : model_name,
            "tasks"    : self._try_get_attr(module, metaclass, "tasks"),
            "mltypes"  : self._try_get_attr(module, metaclass, "mltypes"),
            "model_dir": self._try_get_attr(module, metaclass, "model_dir"),
            "module"   : module,
        }

        self._try_set_attr(module, "arch", arch_name)
        self._try_set_attr(module, "name", model_name)

    # --- Creation ---
    def build(self, name: str, arch: str = None, *args, **kwargs) -> Any:
        """Instantiate a registered model by name and optional architecture.

        Args:
            name: Name of the model to instantiate.
            arch: Optional architecture to narrow the search. Defaults to None.
            kwargs: Arguments to forward to the model's constructor.

        Returns:
            Instance of the registered model.

        Raises:
            ValueError: If the requested model name is not found.
        """
        if not name:
            raise ValueError("Cannot build from an empty name in the 'Models' factory.")

        keys_to_try = [name]
        if self._decamelize:
            keys_to_try.append(depascalize(name))

        # If an architecture is specified, search it first for efficiency.
        if arch:
            arch_key = depascalize(arch) if self._decamelize else arch
            if arch_key in self:
                arch_models = self[arch_key]
                for k in keys_to_try:
                    if k in arch_models:
                        return self._create_instance(arch_models[k]["module"], k, *args, **kwargs)

        # If not found in the specified arch or if no arch was given, search all.
        for arch_models in self.values():
            if not isinstance(arch_models, dict):
                continue
            for k in keys_to_try:
                if k in arch_models:
                    return self._create_instance(arch_models[k]["module"], k, *args, **kwargs)

        raise ValueError(
            f"Model '{name}' is not registered in the '{self._name}' factory. "
            f"Available models: {self.models}."
        )

    # --- Discovery ---
    def search(
        self,
        task: str | None = None,
        mode: str | None = None,
        arch: str | None = None,
    ) -> list[str]:
        """Find all available model names matching a task, mode, and architecture.

        Args:
            task: Task name to filter models. Defaults to None.
            mode: Run mode to filter models. Defaults to None.
            arch: Architecture name to filter models. Defaults to None.

        Returns:
            Sorted list of model names.
        """
        if not any([task, mode, arch]):
            if self.verbose:
                log_error(
                    f"Expected at least one of 'task', 'mode', or 'arch',"
                    f"but got: {task}, {mode}, {arch}."
                )
            return []

        return self.filter(task=task, mode=mode, arch=arch)

    def search_archs(
        self,
        task: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Return available architectures matching a task and mode.

        Args:
            task: Task name to filter architectures. Defaults to None.
            mode: Run mode to filter architectures. Defaults to None.

        Returns:
            Sorted list of architecture names.
        """
        # Get base model list
        models = self.search(task=task, mode=mode)

        # Resolve Architecture names from registry
        flattened_registry = self.flatten_dict
        archs = set()  # Use set to automatically handle duplicates

        for m in models:
            # Check if model exists in registry and has an 'arch' attribute
            entry = flattened_registry.get(m)
            if entry and "arch" in entry:
                a = str(entry["arch"]).strip()
                if a and a.lower() != "none":
                    archs.add(a)

        return sorted(list(archs))

    def find_module(
        self,
        name: str,
        task: str | None = None,
        mode: str | None = None,
        arch: str | None = None,
    ) -> Any | None:
        """Find the module for a model by name and optionally by task, mode,
        and architecture.

        Args:
            name: Model name to find.
            task: Task name to filter models. Defaults to None.
            mode: Run mode to filter models. Defaults to None.
            arch: Architecture name to filter models. Defaults to None.
        """
        available_models = self.filter(task=task, mode=mode, arch=arch)
        for model_name in available_models:
            if model_name == name or (self._decamelize and depascalize(model_name) == name):
                # Retrieve the module from the flattened registry
                flat_registry = self.flatten_dict
                model_entry   = flat_registry.get(model_name)
                if model_entry:
                    return model_entry.get("module")
        return None

    # --- Validation ---
    def has(
        self,
        name: str,
        task: str | None = None,
        mode: str | None = None,
        arch: str | None = None,
    ) -> bool:
        """Check if a model exists by name, task, mode, and architecture.

        Args:
            name: Model name to find.
            task: Task name to filter models. Defaults to None.
            mode: Run mode to filter models. Defaults to None.
            arch: Architecture name to filter models. Defaults to None.
        """
        available_models = self.search(task=task, mode=mode, arch=arch)
        for model_name in available_models:
            if model_name == name or (self._decamelize and depascalize(model_name) == name):
                return True
        return False

    # --- Retrieval ---
    def filter(
        self,
        task: str | None = None,
        mode: str | None = None,
        arch: str | None = None,
    ) -> list[str]:
        """Filter and return all available model names matching a task, mode,
        and architecture.

        Args:
            task: Task name to filter models. If None, return all models.
                Defaults to None.
            mode: Run mode to filter models. If None, return all models.
                Defaults to None.
            arch: Architecture name to filter models. If None, return all models.
                Defaults to None.

        Returns:
            Sorted list of model names.
        """
        # Access the flat registry view
        flatten_models = self.flatten_dict
        models         = list(flatten_models.keys())

        # Filter by Task (e.g., Segmentation, Classification)
        if task and task in Task.values():
            task_enum = Task(task)
            models    = [m for m in models if task_enum in flatten_models[m]["tasks"]]

        # Filter by Mode (e.g., can this model be trained?)
        if mode == "train":
            trainable_types = set(MLType.trainable())
            models = [m for m in models
                      if any(lt in trainable_types for lt in flatten_models[m]["mltypes"])]

        # Filter by Architecture (e.g., resnet50)
        if arch:
            models = [m for m in models if arch == flatten_models[m]["arch"]]

        return sorted(models)


class WeightsFactory(Factory):
    """Factory specialized for organizing weights enums by model name.

    The registered objects are of the type ``WeightsEnum``.
    """

    @property
    def weights_objs(self) -> list[Weights]:
        """Return a list of all registered weights objects."""
        weights_objs = []
        for w_enum in self.values():
            weights_objs.extend(w_enum.values())
        return weights_objs

    # --- Discovery ---
    def find(self, path: Path | str) -> WeightsEnum | None:
        """Find the ``WeightsEnum`` object for a given path.

        Args:
            path: Path to look for the weights object.

        Returns:
            ``WeightsEnum`` object if found, None otherwise.
        """
        path = Path(path).normalize() if path else None

        # Global Discovery
        if path:
            for w_enum in self.values():
                for w in w_enum.values():
                    if path == w.path:
                        return w_enum(w)

        # Return None if no match is found
        return None

    def find_weights_objs(self, path: Path | str) -> Weights | None:
        """Find the ``Weights`` object for a given path.

        Args:
            path: Path to look for the weights object.

        Returns:
            ``Weights`` object if found, None otherwise.
        """
        path = Path(path).normalize() if path else None

        # Global Discovery
        if path:
            for w in self.weights_objs:
                if path == w.path:
                    return w

        # Return None if no match is found
        return None

    # --- Validation ---
    def has(self, path: Path | str) -> bool:
        """Check there is a weights enum registered for a given path.

        Args:
            path: Path to look for the weights object.
        """
        return self.find(path) is not None


# endregion


# ==============================================================================
# region CONSTANTS
# ==============================================================================

ALBUMENTATIONS =        Factory(name="Albumentations")
DATASETS       = DatasetFactory(name="Datasets",  decamelize=True)
BACKBONES      =        Factory(name="Backbones", decamelize=True)
MODELS         =   ModelFactory(name="Models",    decamelize=True)
WEIGHTS        = WeightsFactory(name="Weights",   decamelize=True)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
