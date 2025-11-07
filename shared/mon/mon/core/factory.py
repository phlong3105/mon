#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements factory classes for registering and building objects.
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
    """Base factory class for registering and building objects.

    Args:
        name: Factory's name.
        mapping: Pre-defined ``dict`` of registered classes. Default: ``None``.
        decamelize: If ``True``, converts class names to lowercase with
            underscores. Default: ``False``.
    
    Raises:
        ValueError: If ``name`` is ``None`` or empty.
    """
    
    def __init__(self, name: str, mapping: dict = None, decamelize: bool = False):
        if not name:
            raise ValueError("``name`` must not be empty.")
        self.name       = name
        self.decamelize = decamelize
        super().__init__(mapping or {})
    
    def __repr__(self) -> str:
        """Returns a ``str`` representation of a factory."""
        return f"{self.__class__.__name__}(name={self.name}, items={self})"
    
    def register(self, name: str = None, module: Any = None, replace: bool = False) -> Callable:
        """Registers a class with an optional decorator.

        Args:
            name: Registering name. Default: ``None`` means inferred from the
                class name.
            module: The class to register. Default: ``None``.
            replace: If ``True``, overwrites existing entry. Default: ``False``.

        Returns:
            A decorator if ``module`` is ``None``, else registers directly.

        Raises:
            TypeError: If ``name`` is not a ``str`` or ``None``.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"``name`` must be str or None, got {type(name).__name__}.")
        
        def _register(cls):
            self.register_module(module=cls, name=name, replace=replace)
            return cls
        
        return _register(module) if module else _register
    
    def register_module(self, module: Any, name: str = None, replace: bool = False):
        """Registers a class to the factory.

        Args:
            module: The class to register.
            name: Registering name. Default: ``None`` means inferred from the
                class name.
            replace: If ``True``, overwrites existing entry. Default: ``False``.

        Raises:
            ValueError: If ``module`` is not a class.
        """
        if not inspect.isclass(module):
            raise ValueError(f"``module`` must be a class, got {type(module).__name__}.")
        
        key = (
            name
            or depascalize(module.__name__) if self.decamelize else module.__name__
        )
        if replace or key not in self:
            self[key] = module
    
    def sort(self, reverse: bool = False):
        """Sorts the factory by keys."""
        sorted_items = sorted(self.items(), key=lambda item: item[0], reverse=reverse)
        self.clear()
        self.update(sorted_items)
    
    def build(self, name: str, **kwargs) -> Any:
        """Builds an instance of a registered class.

        Args:
            name: The building class name.
            kwargs: Additional arguments to pass to the class constructor.
           
        Returns:
            A registered class instance or ``None``.

        Raises:
            ValueError: If ``name`` is not in the registry.
        """
        if not name:
            error_console.log(f"``name`` must be defined to build an instance of {self.name}.")
            return None
            
        for k in [name, depascalize(name), pascalize(name)]:
            if name in self:
                instance = self[k](**kwargs)
                if not hasattr(instance, "name"):
                    instance.name = depascalize(k)
                return instance
        raise ValueError(f"``name={name}`` must be in registry.")
    

# ----- Model Factory -----
class ModelFactory(Factory):
    """Factory class for registering and building deep learning models.

    Notes:
        Inherits from ``Factory`` and organizes models by architecture.

    Example:
        >>> MODEL = ModelFactory("Model")
        >>> @MODEL.register(arch="resnet", name="resnet")
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODEL.build(name="resnet", config={})
    """
    
    @property
    def archs(self) -> list[str]:
        """Returns a ``list`` of registered architecture names."""
        return list(self)
    
    @property
    def models(self) -> list[str]:
        """Returns a ``list`` of all registered model names."""
        return [
            model for models in self.values()
            if isinstance(models, dict)
            for model in models
        ]
    
    @property
    def flatten_dict(self) -> dict:
        """Return a flattened ``dict`` of model names as keys."""
        return {
            k2: {**v2, "arch": k1} if isinstance(v2, dict) else v2
            for k1, v1 in self.items()
            for k2, v2 in v1.items()
        }
    
    def register(
        self,
        name   : str  = None,
        arch   : str  = None,
        module : Any  = None,
        replace: bool = False,
    ) -> Callable[[type], type]:
        """Registers a model with an optional decorator.

        Args:
            name: Model name. Default: ``None`` means inferred from the
                model class name.
            arch: Arch name. Default: ``None`` means inferred from the
                model class name.
            module: Model class to register. Default: ``None``.
            replace: If ``True``, overwrites entry. Default: ``False``.

        Returns:
            Decorator if ``module`` is ``None``, else registers directly.

        Raises:
            TypeError: If ``name`` is not a ``str`` or ``None``.
        """
        if name and not isinstance(name, str):
            raise TypeError(f"``name`` must be str or None, got {type(name).__name__}.")
        
        def _register(cls: type) -> type:
            self.register_module(cls, name, arch, replace)
            return cls
        
        return _register(module) if module else _register
    
    def register_module(
        self,
        module : Any,
        name   : str  = None,
        arch   : str  = None,
        replace: bool = False
    ):
        """Registers a model class under an architecture.

        Args:
            name: Model name. Default: ``None`` means inferred from the
                model class name.
            arch: Arch name. Default: ``None`` means inferred from the
                model class name.
            module: Model class to register. Default: ``None``.
            replace: If ``True``, overwrites entry. Default: ``False``.

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
    
    def build(self, name: str = None, arch: str = None, **kwargs):
        """Builds an instance of a registered model.

        Args:
            name: Model name.
            arch: Arch name.
            kwargs: Additional arguments to pass to the class constructor.
           
        Returns:
            A registered model instance or ``None``.

        Raises:
            ValueError: If ``name`` not in the registry.
        """
        arch = arch or name
        if not name:
            error_console.log(f"``name`` must be defined to build an instance of {self.name}.")
            return None
            
        for k in [name, depascalize(name), pascalize(name)]:
            for a, models in self.items():
                if k in models:
                    instance = models[k](**kwargs)
                    if not hasattr(instance, "name"):
                        instance.name = depascalize(k)
                    return instance
        raise ValueError(f"``arch={arch}`` and ``name={name}`` must be in registry.")

    
# ----- Constants -----
ALBUMENTATIONS = Factory(name="Albumentations")
DATASETS       = Factory(name="Datasets", decamelize=True)
MODELS         = ModelFactory(name="Models", decamelize=True)
