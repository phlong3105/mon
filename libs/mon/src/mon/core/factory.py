#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory class registration and instantiation utilities.

This module provides Factory and ModelFactory classes for registering and
instantiating classes by normalized names, along with top-level registries
used throughout the project.
"""

__all__ = [
    "Factory",
    "ModelFactory",
    # Constants
    "ALBUMENTATIONS",
    "DATASETS",
    "MODELS",
]

import inspect
from typing import Any, Callable

from mon.core.console import error_console
from mon.core.utils import depascalize, pascalize


# ==============================================================================
# METAPROGRAMMING & REGISTRATION ENGINE
# ==============================================================================

# --- Base Factory ---
class Factory(dict):
    """A dictionary-backed factory for class registration and construction.

    Use register decorators or direct calls to add classes and build instances
    by name.

    Attributes:
        _name (str): Factory name used in messages and repr.
        _decamelize (bool): Whether registered class names are normalized to
            snake_case.
    """
    
    def __init__(self, name: str, mapping: dict = None, decamelize: bool = False):
        """Initialize the factory.

        Args:
            name: Factory name.
            mapping: Optional initial mapping of registered classes.
            decamelize: If True, normalize class names to snake_case.

        Raises:
            ValueError: If ``name`` is empty.
        """
        if not name:
            raise ValueError("``name`` must not be empty.")
        
        self._name       = name
        self._decamelize = decamelize
        
        super().__init__(mapping or {})
    
    # --- Magic Methods ---
    def __repr__(self) -> str:
        """Return a string representation of the factory."""
        return f"{self.__class__.__name__}(name={self._name}, items={self})"
    
    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the factory name."""
        return self._name
    
    # --- Register ---
    def register(self, name: str = None, module: Any = None, replace: bool = False) -> Callable:
        """Register a class or return a decorator for registration.

        Args:
            name: Optional registration key.
            module: Class to register immediately.
            replace: Overwrite existing entry when True.

        Returns:
            A decorator or the registered class.

        Raises:
            TypeError: If ``name`` is not a str or None.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"``name`` must be str or None, got {type(name)}.")
        
        def _register(cls):
            self._register_module(module=cls, name=name, replace=replace)
            return cls
        
        return _register(module) if module else _register
    
    def _register_module(self, module: Any, name: str = None, replace: bool = False):
        """Register a class under a registry key.

        Args:
            module: The class to register.
            name: Optional key; inferred when None.
            replace: Overwrite an existing entry when True.

        Raises:
            ValueError: If ``module`` is not a class.
        """
        if not inspect.isclass(module):
            raise ValueError(f"``module`` must be a class, got {type(module)}.")
        
        key = (
            name
            or depascalize(module.__name__) if self._decamelize else module.__name__
        )
        if replace or key not in self:
            self[key] = module
    
    def sort(self, reverse: bool = False):
        """Sort registry entries by key.

        Args:
            reverse: Sort in descending order when True.
        """
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        self.clear()
        self.update(sorted_items)
    
    # --- Build ---
    def build(self, name: str, **kwargs) -> Any:
        """Instantiate a registered class by name.

        Try normalized name variants (snake or pascal case) when looking up.

        Args:
            name: Registered key to instantiate.
            kwargs: Forwarded to the class constructor.

        Returns:
            The instantiated object.

        Raises:
            ValueError: If ``name`` is not present in registry.
        """
        if not name:
            error_console.log(f"``name`` must be defined to build an instance of {self._name}.")
            return None
            
        for k in [name, depascalize(name), pascalize(name)]:
            if name in self:
                instance = self[k](**kwargs)
                if not hasattr(instance, "name"):
                    instance._name = depascalize(k)
                return instance
        raise ValueError(f"``name={name}`` must be in registry.")
    

# --- Specialization ---
class ModelFactory(Factory):
    """A factory specialized for organizing models by architecture.

    Maintain a nested mapping arch -> {model_name: model_class} and provide
    registration helpers.

    Attributes:
        _name (str): Factory name used in messages and repr.
        _decamelize (bool): Whether registered class names are normalized to
            snake_case.
        items (dict): Mapping of architecture name to model name -> model
            class.
    """
    
    # --- Properties ---
    @property
    def archs(self) -> list[str]:
        """Return a list of registered architecture names."""
        return list(self)
    
    @property
    def models(self) -> list[str]:
        """Return a list of all registered model names."""
        return [
            model for models in self.values()
            if isinstance(models, dict)
            for model in models
        ]
    
    @property
    def flatten_dict(self) -> dict:
        """Return a flattened mapping of model name to class with arch metadata."""
        return {
            k2: {**v2, "arch": k1} if isinstance(v2, dict) else v2
            for k1, v1 in self.items()
            for k2, v2 in v1.items()
        }
    
    # --- Register ---
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Register a model class under an architecture or return a decorator.

        Args:
            name: Optional model name.
            arch: Optional architecture name.
            module: Class to register immediately.
            replace: Overwrite existing entry when True.

        Returns:
            Decorator or the registered class.

        Raises:
            TypeError: If ``name`` is not a str or None.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"``name`` must be str or None, got {type(name).__name__}.")
        
        def _register(cls: type) -> type:
            self._register_module(cls, name, arch, replace)
            return cls
        
        return _register(module) if module else _register
    
    def _register_module(
        self,
        module : Any,
        name   : str  = None,
        arch   : str  = None,
        replace: bool = False
    ):
        """Register a model class under an architecture key.

        Args:
            module: Model class to register.
            name: Optional model name; inferred when None.
            arch: Optional architecture name; inferred when None.
            replace: If True, overwrite an existing entry.

        Raises:
            ValueError: If ``module`` is not a class.
        """
        if not inspect.isclass(module):
            raise ValueError(f"``module`` must be a class, got {type(module).__name__}.")
        
        module_key = name or depascalize(module.__name__)
        arch_key   = arch or depascalize(getattr(module, "arch", module.__name__))
        
        if arch_key not in self:
            self[arch_key] = {}
        if replace or module_key not in self[arch_key]:
            self[arch_key][module_key] = module
    
    # --- Build ---
    def build(self, name: str, arch: str = None, **kwargs):
        """Instantiate a registered model by name and optional architecture.

        Args:
            name: Model name to instantiate.
            arch: Optional architecture to restrict lookup.
            kwargs: Forwarded to model constructor.

        Returns:
            The instantiated model.

        Raises:
            ValueError: If the requested ``arch``/``name`` pair is not found.
        """
        arch = arch or name
        if not name:
            error_console.log(f"``name`` must be defined to build an instance of {self._name}.")
            return None
            
        for k in [name, depascalize(name), pascalize(name)]:
            for a, models in self.items():
                if k in models:
                    instance = models[k](**kwargs)
                    if not hasattr(instance, "name"):
                        instance._name = depascalize(k)
                    return instance
        raise ValueError(f"``arch={arch}`` and ``name={name}`` must be in registry.")


# ==============================================================================
# GLOBAL REGISTRIES
# ==============================================================================

# --- Domain Factories ---
ALBUMENTATIONS = Factory(name="Albumentations")
DATASETS       = Factory(name="Datasets", decamelize=True)
MODELS         = ModelFactory(name="Models", decamelize=True)
