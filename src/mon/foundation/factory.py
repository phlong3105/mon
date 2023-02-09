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
from typing import Any, Callable

import humps


# region Factory

class Factory(dict):
    """The base factory class for building arbitrary objects. It registers
    classes to a registry dictionary and then dynamically builds objects of the
    registered classes later.
    
    Notes:
        We inherit Python built-in :class:`dict`.
    
    Args:
        name: The factory's name.
        
    Example:
        >>> MODEL = Factory("Model")
        >>> @MODEL.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODEL.build(name="ResNet", **resnet_hparams)
    """
    
    def __init__(self, name: str, mapping: dict | None = None, *kwargs):
        if name is None or name == "":
            raise ValueError(
                f"name must be given to create a valid factory object."
            )
        mapping    = mapping or {}
        self._name = name
        super().__init__(mapping)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(name={self._name}, items={self})"
    
    @property
    def name(self) -> str:
        """The name of the current :class:`Factory` object."""
        return self._name
    
    def register(
        self,
        name   : str | None = None,
        module : Callable   = None,
        replace: bool       = False,
    ) -> callable:
        """Register a module/class.
        
        Args:
            name: A module/class name. If None, automatically infer from the
                given :param:`module`.
            module: The registering module.
            replace: If True, overwrite the existing module. Defaults to False.
        
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
            raise TypeError(f"name must be a str, but got {type(name)}.")
        
        # Use it as a normal method: x.register(module=SomeClass)
        if module is not None:
            self.register_module(
                module_cls  = module,
                module_name = name,
                replace     = replace
            )
            return module
        
        # Use it as a decorator: @x.register()
        def _register(cls):
            self.register_module(
                module_cls  = cls,
                module_name = name,
                replace     = replace
            )
            return cls
        
        return _register
    
    def register_module(
        self,
        module_cls : Callable,
        module_name: str | None = None,
        replace    : bool       = False
    ):
        """Register a module/class.
        
        Args:
            module_cls: The registering module/class.
            module_name: A module/class name. If None, automatically infer from
                the given :param:`module`.
            replace: If True, overwrite the existing module. Defaults to False.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(
                f"module_cls must be a class interface, but got "
                f"{type(module_name)}."
            )
        
        module_name = module_name or module_cls.__name__
        
        # Register module name using the snake_case format
        snake_name = humps.depascalize(humps.pascalize(module_name))
        if replace or snake_name not in self:
            self[snake_name] = module_cls
        
        # Register module name using the kebab-case format
        kebab_name = humps.kebabize(module_name)
        if replace or kebab_name not in self:
            self[kebab_name] = module_cls
        
        # Register module name using the PascalCase format
        pascal_name = module_cls.__name__
        if replace or pascal_name not in self:
            self[pascal_name] = module_cls
    
    def build(
        self,
        name   : str  | None = None,
        cfg    : dict | None = None,
        to_dict: bool        = False,
        **kwargs
    ):
        """Build an instance of the registered class corresponding to the given
        name.
        
        Args:
            name: A class name.
            cfg: The class's arguments.
            to_dict: If True, return a dictionary of
                {:param:`name`: attr:`instance`}. Defaults to False.
            
        Returns:
            An instance of the registered class.
        """
        if (name is None and cfg is None) \
            or (name is None and cfg is not None and "name" not in cfg):
            return None
        if cfg is not None:
            cfg_    = copy.deepcopy(cfg)
            name    = name or cfg_.pop("name", None)
            kwargs |= cfg_
        if name is None or name not in self:
            raise ValueError(
                "name must be a valid keyword inside the registry."
            )
        
        obj = self[name](**kwargs)
        if getattr(obj, "name", None) is None:
            obj.name = humps.depascalize(humps.pascalize(name))
        
        if to_dict:
            return {f"{name}": obj}
        else:
            return obj
    
    def build_instances(
        self,
        cfgs   : list[Any] | None,
        to_dict: bool = False,
        **kwargs
    ):
        """Build multiple instances of different classes with the given
        arguments.
        
        Args:
            cfgs: A list of classes' arguments. Each item can be:
                - A name (string).
                - A dictionary of arguments containing the 'name' key.
            to_dict: If True, return a dictionary of
                {:param:`name`: attr:`instance`}. Defaults to False.
                
        Returns:
            A list, or a dictionary of instances.
        """
        if cfgs is None:
            return None
        if not isinstance(cfgs, list):
            raise ValueError(f"cfgs must be a list, but got {type(cfgs)}.")
        
        cfgs_ = copy.deepcopy(cfgs)
        objs = {} if to_dict else []
        for cfg in cfgs_:
            if isinstance(cfg, str):
                name = cfg
            elif isinstance(cfg, dict):
                name    = cfg.pop("name")
                kwargs |= cfg
            else:
                raise ValueError(
                    f"item inside cfgs must be a str or dict, but got "
                    f"{type(cfg)}."
                )
            
            obj = self.build(name=name, to_dict=to_dict, **kwargs)
            if obj is not None:
                if to_dict:
                    objs |= obj
                else:
                    objs.append(obj)
        
        return objs if len(objs) > 0 else None

# endregion
