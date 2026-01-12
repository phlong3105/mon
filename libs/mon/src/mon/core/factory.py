#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class registration and instantiation utilities.

This module provides Factory and ModelFactory classes for registering and
instantiating classes.
"""

from __future__ import annotations

__all__ = [
    "Factory",
    "ArchModelFactory",
    # Constants
    "ALBUMENTATIONS",
    "BACKBONES",
    "DATASETS",
    "MODELS",
    "WEIGHTS",
]

import inspect
from typing import Any, Callable

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
    def __init__(self, name: str, mapping: dict = None, decamelize: bool = False):
        """Initialize a new instance.

        Args:
            name: Name for the factory.
            mapping: Optional initial dictionary of registered classes.
                Defaults to None.
            decamelize: If True, normalize class names to snake_case.
                Defaults to False.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError(f"Expected 'name' to be a non-empty string, but got '{name}'.")
        
        self._name       = name
        self._decamelize = decamelize
        super().__init__(mapping or {})
    
    # --- Representation ---
    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"{self.__class__.__name__}(name='{self._name}', items={list(self.items())})"
    
    # --- Mathematical Operators ---
    def sort(self, reverse: bool = False):
        """Sort the registry entries alphabetically by key.

        Args:
            reverse: If True, sort in descending order. Defaults to False.
        """
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        self.clear()
        self.update(sorted_items)
    
    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the factory's name."""
        return self._name
    
    # --- Register ---
    def register(self, name: str = None, module: Any = None, replace: bool = False) -> Callable:
        """Register a class or return a decorator for registration.

        Args:
            name: Optional name to register the class under. Defaults to None.
            module: Class to register immediately. Defaults to None.
            replace: If True, overwrite any existing registration for the name.
                Defaults to False.

        Returns:
            Decorator if ``module`` is None, otherwise the registered class.
        """
        def _register(cls):
            self._register_module(module=cls, name=name, replace=replace)
            return cls
        
        return _register(module) if module is not None else _register
    
    def _register_module(self, module: Any, name: str = None, replace: bool = False):
        """Register a class internally.

        Args:
            module: Class or function to register.
            name: Optional key to register under. Defaults to None.
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
            raise KeyError(f"'{key}' is already registered in the '{self._name}' factory.")

        self[key] = module
        self._try_set_name(module, key)

    # --- Build ---
    def build(self, name: str, **kwargs) -> Any:
        """Instantiate a registered class by name.

        Args:
            name: Registered key of the class to instantiate.
            kwargs: Arguments to forward to the class constructor.

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

        return self._create_instance(self[key], name, **kwargs)
    
    def _create_instance(self, cls: type, name: str, **kwargs) -> Any:
        """Instantiate and tag the object with its registered name internally."""
        instance = cls(**kwargs)
        self._try_set_name(instance, name)
        return instance
    
    # --- Utils ---
    @staticmethod
    def _try_set_name(module: Any, name: str):
        """Attempt to set the ``name`` and ``_name`` attributes on the module."""
        if inspect.isclass(module):
            try:
                if not hasattr(module, "name") or getattr(module, "name") != name:
                    module.name = name
                if not hasattr(module, "_name") or getattr(module, "_name") != name:
                    module._name = name
            except (AttributeError, TypeError):
                # Silently fail if the attribute is not settable (e.g., on built-ins).
                pass
    

# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class ArchModelFactory(Factory):
    """Factory specialized for organizing models by architecture.

    Maintain a nested structure: ``arch -> {model_name: model_class}``. Provide
    specialized methods for registering and building models within this
    structure.
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
                if isinstance(model_data, dict):
                    flat_dict[model_name] = {**model_data, "arch": arch}
                else:
                    flat_dict[model_name] = {"module": model_data, "arch": arch}
        return flat_dict
    
    # --- Register ---
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        variant: str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Register a model class or return a decorator for registration.

        Args:
            name: Full model name. Defaults to None.
            arch: Architecture name. Defaults to None.
            variant: Model variant name. Defaults to None.
            module: Class to register immediately. Defaults to None.
            replace: If True, overwrite any existing registration.
                Defaults to False.

        Returns:
            Decorator or the registered class.
        """
        def _register(cls: type) -> type:
            self._register_module(cls, name, arch, variant, replace)
            return cls
        
        return _register(module) if module is not None else _register
    
    def _register_module(
        self,
        module : Any,
        name   : str  = None,
        arch   : str  = None,
        variant: str  = None,
        replace: bool = False
    ):
        """Register a model class internally.

        Args:
            module: Class or function to register.
            name: Full model name. Defaults to None.
            arch: Architecture name. Defaults to None.
            variant: Model variant name. Defaults to None.
            replace: If True, overwrite an existing entry. Defaults to False.

        Raises:
            TypeError: If ``module`` is not a class or function.
        """
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                f"Expected 'module' to be a class or function, but got {type(module).__name__}."
            )
        
        arch_name  = arch or getattr(module, "_arch", getattr(module, "arch", module.__name__))
        model_name = name or getattr(module, "_name", getattr(module, "name", f"{arch_name}_{variant}" if variant else arch_name))
        
        if self._decamelize:
            arch_name  = depascalize(arch_name)
            model_name = depascalize(model_name)
        
        self.setdefault(arch_name, {})

        if not replace and model_name in self[arch_name]:
            # The model is already registered; silently return to avoid errors.
            return

        self[arch_name][model_name] = module
        self._try_set_name(module, model_name)

    # --- Build ---
    def build(self, name: str, arch: str = None, **kwargs) -> Any:
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
                        return self._create_instance(arch_models[k], k, **kwargs)
        
        # If not found in the specified arch or if no arch was given, search all.
        for arch_models in self.values():
            if not isinstance(arch_models, dict):
                continue
            for k in keys_to_try:
                if k in arch_models:
                    return self._create_instance(arch_models[k], k, **kwargs)
        
        raise ValueError(
            f"Model '{name}' is not registered in the '{self._name}' factory. "
            f"Available models: {self.models}."
        )

# endregion


# ==============================================================================
# region CONSTANTS
# ==============================================================================

ALBUMENTATIONS = Factory(name="Albumentations")
DATASETS       = Factory(name="Datasets",  decamelize=True)
BACKBONES      = Factory(name="Backbones", decamelize=True)
MODELS         = ArchModelFactory(name="Models",  decamelize=True)
WEIGHTS        = ArchModelFactory(name="Weights", decamelize=True)

# endregion
