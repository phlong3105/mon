#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for factory classes.

This module provides factory classes for registering and building various
objects, such as deep learning models and datasets. It includes a base
Factory class, and a specialized ModelFactory for organizing models by
architecture.
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


# ----- Base Factory -----
class Factory(dict):
    """A base factory class for registering and building classes.
    
    This class extends the built-in ``dict`` to provide functionality for
    registering classes with optional decorators and building instances of
    those classes. It supports optional name transformations for registration.
    """
    
    def __init__(self, name: str, mapping: dict = None, decamelize: bool = False):
        """Initializes the factory.
        
        Args:
            name (str): Factory's name.
            mapping (dict): Pre-defined dictionary of registered classes.
                Defaults to None.
            decamelize (bool): If True, converts class names to lowercase with
                underscores. Defaults to False.
        
        Raises:
            ValueError: If ``name`` is ``None`` or empty.
        """
        if not name:
            raise ValueError("``name`` must not be empty.")
        
        self._name       = name
        self._decamelize = decamelize
        
        super().__init__(mapping or {})
    
    # ----- Magic Methods -----
    def __repr__(self) -> str:
        """Returns a string representation of the factory object.
        
        Returns:
            str: String representation.
        """
        return f"{self.__class__.__name__}(name={self._name}, items={self})"
    
    # ----- Properties -----
    @property
    def name(self) -> str:
        """Getter for the factory name.
        
        Returns:
            str: The factory name.
        """
        return self._name
    
    # ----- Register -----
    def register(self, name: str = None, module: Any = None, replace: bool = False) -> Callable:
        """Registers a class to the factory with an optional decorator.

        Args:
            name (str, optional): The name of the class to register.
                Defaults to None.
            module (Any, optional): The class to register. Defaults to None.
            replace (bool, optional): If True, overwrites existing entry.
                Defaults to False.

        Returns:
            Callable: Decorator if ``module`` is None, else registers directly.

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
        """Registers a class to the factory.

        Args:
            module (Any): The class to register.
            name (str, optional): The name of the class to register.
                Defaults to None means inferred from the class name.
            replace (bool, optional): If True, overwrites existing entry.
                Defaults to False.

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
        """Sorts the factory by keys."""
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        self.clear()
        self.update(sorted_items)
    
    # ----- Build -----
    def build(self, name: str, **kwargs) -> Any:
        """Builds an instance of a registered class.

        Args:
            name (str): The name of the class to build.
            kwargs: Additional arguments to pass to the class constructor.
           
        Returns:
            A registered class instance or None.
        
        Raises:
            ValueError: If ``name`` is not in the registry.
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
    

# ----- Model Factory -----
class ModelFactory(Factory):
    """A factory class for registering and building machine learning models
    organized by architecture.

    This class extends the base Factory class to provide functionality for
    registering models under specific architectures. It allows for easy
    organization and retrieval of models based on their architecture type.

    Example:
        >>> MODEL = ModelFactory("Model")
        >>> @MODEL.register(arch="resnet", name="resnet")
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODEL.build(name="resnet", config={})
    """
    
    # ----- Properties -----
    @property
    def archs(self) -> list[str]:
        """Getter for all registered architecture names.
        
        Returns:
            list[str]: A list of all registered architecture names.
        """
        return list(self)
    
    @property
    def models(self) -> list[str]:
        """Getter for all registered model names.
        
        Returns:
            list[str]: A list of all registered model names.
        """
        return [
            model for models in self.values()
            if isinstance(models, dict)
            for model in models
        ]
    
    @property
    def flatten_dict(self) -> dict:
        """Getter for a flattened dictionary of all registered models.
        
        Returns:
            dict: A flattened dictionary where each key is a model name, and
                the value is the corresponding class with an added "arch" key.
        """
        return {
            k2: {**v2, "arch": k1} if isinstance(v2, dict) else v2
            for k1, v1 in self.items()
            for k2, v2 in v1.items()
        }
    
    # ----- Register -----
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Registers a model class under an architecture with an optional
        decorator.

        Args:
            name (str, optional): Model name. Defaults to None means inferred
                from the model class name.
            arch (str, optional): Arch name. Defaults to None means inferred
                from the model class name.
            module (Any, optional): Model class to register. Defaults to None.
            replace (bool, optional): If True, overwrites entry. Defaults to
                False.

        Returns:
            Decorator if ``module`` is None, else registers directly.

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
        """Registers a model class under an architecture.

        Args:
            name (str, optional): Model name. Defaults to None means inferred
                from the model class name.
            arch (str, optional): Arch name. Defaults to None means inferred
                from the model class name.
            module (Any): Model class to register.
            replace (bool, optional): If True, overwrites entry. Defaults to
                False.

        Raises:
            ValueError: If ``module_cls`` is not a class.
        """
        if not inspect.isclass(module):
            raise ValueError(f"``module`` must be a class, got {type(module).__name__}.")
        
        module_key = name or depascalize(module.__name__)
        arch_key   = arch or depascalize(getattr(module, "arch", module.__name__))
        
        if arch_key not in self:
            self[arch_key] = {}
        if replace or module_key not in self[arch_key]:
            self[arch_key][module_key] = module
    
    # ----- Build -----
    def build(self, name: str, arch: str = None, **kwargs):
        """Builds an instance of a registered model class.

        Args:
            name (str, optional): Model name.
            arch (str, optional): Arch name. Defaults to None means inferred
                from the model name.
            kwargs: Additional arguments to pass to the model constructor.
           
        Returns:
            A registered model instance or None.

        Raises:
            ValueError: If ``name`` not in the registry.
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

    
# ----- Constants -----
ALBUMENTATIONS = Factory(name="Albumentations")
DATASETS       = Factory(name="Datasets", decamelize=True)
MODELS         = ModelFactory(name="Models", decamelize=True)
