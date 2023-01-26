#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a factory method design pattern. It defines mechanisms
for registering classes and dynamically build them at run-time.
"""

from __future__ import annotations

__all__ = [
    "Factory",
]

import copy
import inspect
from typing import TYPE_CHECKING

import humps
import munch

if TYPE_CHECKING:
    from mon.core.typing import CallableType


# region Factory

class Factory:
    """The base factory class for building arbitrary objects. It registers
    classes to a registry dictionary and then dynamically builds objects of the
    registered classes later.
    
    Args:
        name: The factory's name.
        
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
    """
    
    def __init__(self, name: str):
        self._name     = name
        self._registry = {}
    
    def __len__(self) -> int:
        """Return the number of items stored inside :attr:`registry`."""
        return len(self._registry)
    
    def __contains__(self, key: str) -> bool:
        """Return True if a key is in the :attr:`registry`. Otherwise,
        return False.
        
        Args:
            key: The search key, in other words, the name of the class.
        """
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return self.__class__.__name__ \
            + f"(name={self._name}, items={self._registry})"
        
    @property
    def name(self) -> str:
        """The name of the current :class:`Factory` object."""
        return self._name
    
    @property
    def registry(self) -> dict:
        """A dictionary containing keys (classes' names) and registered
        classes.
        """
        return self._registry
    
    def get(self, key: str) -> CallableType | None:
        """Get the class corresponding to a given key. If the key doesn't exist
        in :attr:`register`, return None.
        
        Args:
            key: The name of the registered class.
        """
        if key in self._registry:
            return self._registry[key]
        return None
        
    def register(
        self,
        name  : str          | None = None,
        module: CallableType | None = None,
        force : bool                = False,
    ) -> callable:
        """Register a module/class.
        
        Args:
            name: A module/class name. If None, automatically infer from the
                given :param:`module`.
            module: The registering module.
            force: If True, overwrites the existing module. Defaults to False.
        
        Example:
            # >>> backbones = Factory("backbone")
            # >>>
            # >>> @backbones.register()
            # >>> class ResNet:
            # >>>     pass
            # >>>
            # >>> @backbones.register(name="mnet")
            # >>> class MobileNet:
            # >>>     pass
            # >>>
            # >>> class ResNet:
            # >>>     pass
            # >>> backbones.register(ResNet)
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                f":param:`name` must be `None` or `str`. But got: {type(name)}."
            )
        
        # Use it as a normal method: x.register(module=SomeClass)
        if module is not None:
            self.register_module(module, name, force)
            return module
        
        # Use it as a decorator: @x.register()
        def _register(cls):
            self.register_module(cls, name, force)
            return cls
        
        return _register
    
    def register_module(
        self,
        module_class: CallableType,
        module_name : str | None = None,
        force       : bool       = False
    ):
        """Register a module/class.
        
        Args:
            module_class: The registering module/class.
            module_name: A module/class name. If None, automatically infer from
                the given :param:`module`.
            force: If True, overwrites the existing module. Defaults to False.
        """
        assert inspect.isclass(module_class)
       
        if module_name is None:
            module_name = module_class.__name__
        # module_name = to_list(module_name)

        # Register module name using the snake_case format
        if not humps.is_snakecase(module_name):
            module_name = humps.decamelize(module_name)
            module_name = module_name.lower()
        if force or module_name not in self._registry:
            self._registry[module_name] = module_class
        else:
            # logger.debug(f"{name} is already registered in {self.name}.")
            # continue
            return
        
        # Register module name using the PascalCase format
        if not humps.is_pascalcase(module_name):
            module_name = module_class.__name__
        if force or module_name not in self._registry:
            self._registry[module_name] = module_class
        else:
            # logger.debug(f"{name} is already registered in {self.name}.")
            # continue
            return

    def build(self, name: str | None = None, cfg: dict | None = None, **kwargs):
        """Build an instance of the registered class corresponding to the given
        name.
        
        Args:
            name: A class name.
            cfg: The class's arguments.
        
        Returns:
            An instance of the registered class.
        """
        if name is None and cfg is None:
            return None
        if name is None and cfg is not None:
            assert "name" in cfg
            cfg_    = copy.deepcopy(cfg)
            name_   = cfg_.pop("name")
            name    = name or name_
            kwargs |= cfg_
       
        assert name is not None
        assert name in self.registry
        if name not in self.registry:
            return None
        
        name     = humps.camelize(name) if humps.is_snakecase(name) else name
        instance = self.registry[name](**kwargs)
        if not hasattr(instance, "name"):
            instance.name = name
        return instance
    
    def build_instances(self, cfgs: list[dict | munch.Munch] | None, **kwargs):
        """Build multiple instances of different classes with the given
        arguments.

        Args:
            cfgs: A list of classes' arguments. Each item can be:
                - A name (string)
                - A dictionary of arguments containing the “name” key.

        Returns:
            A list of instances.
        """
        if cfgs is None:
            return None
        assert isinstance(cfgs, list)
        
        cfgs_     = copy.deepcopy(cfgs)
        instances = []
        for cfg in cfgs_:
            if isinstance(cfg, str):
                name    = cfg
                instances.append(self.build(name=name, **kwargs))
            else:
                name    = cfg.pop("name")
                kwargs |= cfg
            instances.append(self.build(name=name, **kwargs))
        
        return instances if len(instances) > 0 else None

# endregion
