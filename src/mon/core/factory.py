#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Factory Module.

This module implements a factory method design pattern. It defines mechanisms
for registering classes and dynamically build them at run-time.
"""

from __future__ import annotations

__all__ = [
    "Factory",
    "ModelFactory",
]

import copy
import inspect
from typing import Any

import humps

from mon.core.typing import _callable


# region Factory

class Factory(dict):
    """The base factory class for building arbitrary objects. It registers
    classes to a registry :obj:`dict` and then dynamically builds objects of
    the registered classes later.
    
    Notes:
        We inherit Python built-in :obj:`dict`.
    
    Args:
        name: The factory's name.
        
    Example:
        >>> MODEL = Factory("Model")
        >>> @MODEL.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet = MODEL.build(name="ResNet", **resnet_hparams)
    """
    
    def __init__(self, name: str, mapping: dict = None, *args, **kwargs):
        if name in [None, "None", ""]:
            raise ValueError(f"`name` must be given to create a valid factory "
                             f"object.")
        mapping   = mapping or {}
        self.name = name
        super().__init__(mapping)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(name={self.name}, items={self})"
    
    def register(
        self,
        name   : str       = None,
        module : _callable = None,
        replace: bool      = False,
    ) -> callable:
        """Register a module/class.
        
        Args:
            name: A module/class name. If ``None``, automatically infer from the
                given :obj:`module`.
            module: The registering module.
            replace: If ``True``, overwrite the existing module.
                Default: ``False``.
        
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
            raise TypeError(f"`name` must be a `str`, but got {type(name)}.")
            
        # Use it as a normal method: x.register(module=SomeClass)
        if module:
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
        module_cls : _callable,
        module_name: str  = None,
        replace    : bool = False
    ):
        """Register a module/class.
        
        Args:
            module_cls: The registering module/class.
            module_name: A module/class name. If ``None``, automatically infer
                from the given :obj:`module`.
            replace: If ``True``, overwrite the existing module.
                Default: ``False``.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"`module_cls` must be a class interface, but got "
                             f"{type(module_name)}.")
        
        module_name = module_name or humps.kebabize(module_cls.__name__)
        if replace or module_name not in self:
            self[module_name] = module_cls
        
    def build(
        self,
        name   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ):
        """Build an instance of the registered class corresponding to the given
        name.
        
        Args:
            name: A class name.
            config: The class's arguments.
            to_dict: If ``True``, return a :obj:`dict` of
                {:obj:`name` : :obj:`instance`}. Default: ``False``.
            
        Returns:
            An instance of the registered class.
        """
        if (
            (name is None and config is None) or
            (name is None and config and "name" not in config)
        ):
            return None
        if config:
            config_ = copy.deepcopy(config)
            name    = name or config_.pop("name", None)
            kwargs |= config_
            
        # Loop through all possible naming conventions
        if name:
            kebab_name  = humps.kebabize(name)
            snake_name  = humps.depascalize(humps.pascalize(name))
            pascal_name = humps.pascalize(name)
            for n in [name, kebab_name, snake_name, pascal_name]:
                if n in self:
                    name = n
        if name is None or name not in self:
            raise ValueError(
                f"`name` must be a valid keyword inside the registry, "
                f"but got {name}."
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
        configs: list[Any],
        to_dict: bool = False,
        **kwargs
    ):
        """Build multiple instances of different classes with the given
        :obj:`args`.
        
        Args:
            configs: A list of classes' arguments. Each item can be:
                - A name (:obj:`str`).
                - A dictionary of arguments containing the ``'name'`` key.
            to_dict: If ``True``, return a :obj:`dict` of
                {:obj:`name`: :obj:`instance`}. Default: ``False``.
                
        Returns:
            A list, or a dictionary of instances.
        """
        if configs is None:
            return None
        if not isinstance(configs, list):
            raise ValueError(f"`configs` must be a `list`, "
                             f"but got {type(configs)}.")
        
        configs_ = copy.deepcopy(configs)
        objs     = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name = config
            elif isinstance(config, dict):
                name = config.pop("name")
                # kwargs |= config
            else:
                raise ValueError(f"Item inside `configs` must be a `str` or "
                                 f"`dict`, but got {type(config)}.")
            
            obj = self.build(name=name, to_dict=to_dict, **config)
            if obj:
                if to_dict:
                    objs |= obj
                else:
                    objs.append(obj)
        
        return objs if len(objs) > 0 else None


class ModelFactory(Factory):
    """TThe factory for registering and building models.
    
    Notes:
        We inherit Python built-in :obj:`dict`.
    
    Example:
        >>> MODEL = ModelFactory("Model")
        >>> @MODEL.register(arch="resnet", name="resnet")
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODEL.build(name="resnet", **resnet_hparams)
    """
    
    @property
    def archs(self) -> list[str]:
        return list(self.keys())
    
    @property
    def models(self) -> list[str]:
        return [sub_k for k, d in self.items() if isinstance(d, dict) for sub_k in d]
    
    def register(
        self,
        name   : str       = None,
        arch   : str       = None,
        module : _callable = None,
        replace: bool      = False,
    ) -> callable:
        """Register a model.
        
        Args:
            name: Model's name. If ``None``, automatically infer from the given
                :obj:`module`.
            arch: Architecture's name. If ``None``, automatically infer from
                the given :obj:`module`.
            module: The registering module.
            replace: If ``True``, overwrite the existing module.
                Default: ``False``.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"`name` must be a `str`, but got {type(name)}.")
        
        # Use it as a normal method: x.register(module=SomeClass)
        if module:
            self.register_module(
                module_cls  = module,
                module_name = name,
                arch_name   = arch,
                replace     = replace
            )
            return module
        
        # Use it as a decorator: @x.register()
        def _register(cls):
            self.register_module(
                module_cls  = cls,
                module_name = name,
                arch_name   = arch,
                replace     = replace
            )
            return cls
        
        return _register
    
    def register_module(
        self,
        module_cls : _callable,
        module_name: str  = None,
        arch_name  : str  = None,
        replace    : bool = False
    ):
        """Register a module/class.
        
        Args:
            module_cls: The registering module/class.
            module_name: Module/class name. If ``None``, automatically infer
                from the given :obj:`module`.
            arch_name: Architecture's name. If ``None``, automatically infer
                from the given :obj:`module`.
            replace: If ``True``, overwrite the existing module.
                Default: ``False``.
        """
        if not inspect.isclass(module_cls):
            raise ValueError(f"`module_cls` must be a class interface, "
                             f"but got {type(module_name)}.")
        
        module_name = module_name or humps.kebabize(module_cls.__name__)
        arch_name   = arch_name   or humps.kebabize(getattr(module_cls, "arch", None))
        arch_name   = arch_name   or humps.kebabize(module_cls.__name__)
        if arch_name not in self:
            self[arch_name] = {}
        if replace or module_name not in self[arch_name]:
            self[arch_name][module_name] = module_cls
    
    def build(
        self,
        name   : str  = None,
        arch   : str  = None,
        config : dict = None,
        to_dict: bool = False,
        **kwargs
    ):
        """Build an instance of the registered model's variant corresponding to
        the given name.
        
        Args:
            name: Model's name.
            arch: Architecture's name.
            config: The class's arguments.
            to_dict: If ``True``, return a :obj:`dict` of
                {:obj:`name` : attr:`instance`}. Default: ``False``.
            
        Returns:
            An instance of the registered class.
        """
        if (
            (name is None and config is None) or
            (name is None and config and "name" not in config)
        ):
            return None
        if config:
            config_  = copy.deepcopy(config)
            name     = name or config_.pop("name", None)
            kwargs  |= config_
        arch = arch or name
        
        # Loop through all possible naming conventions
        if name:
            kebab_name  = humps.kebabize(name)
            snake_name  = humps.depascalize(humps.pascalize(name))
            pascal_name = humps.pascalize(name)
            for n in [name, kebab_name, snake_name, pascal_name]:
                for a, models_dict in self.items():
                    if n in models_dict:
                        name = n
                        arch = a if arch != a else arch
                        break
        if (
            arch is None and arch not in self
            or name is None and name not in self[arch]
        ):
            raise ValueError(f"`arch` and `name` must be a valid keyword inside "
                             f"the registry, but got {arch} and {name}.")
        
        obj = self[arch][name](**kwargs)
        if getattr(obj, "name", None) is None:
            obj.name = humps.depascalize(humps.pascalize(name))
          
        if to_dict:
            return {f"{name}": obj}
        else:
            return obj
    
    def build_instances(
        self,
        configs: list[Any],
        to_dict: bool = False,
        **kwargs
    ):
        """Build multiple instances of different classes with the given
        :obj:`args`.
        
        Args:
            configs: A list of classes' arguments. Each item can be:
                - A name (:obj:`str`).
                - A dictionary of arguments containing the ``'name'`` key.
            to_dict: If ``True``, return a :obj:`dict` of
                {:obj:`name`: attr:`instance`}. Default: ``False``.
                
        Returns:
            A list, or a dictionary of instances.
        """
        if configs is None:
            return None
        if not isinstance(configs, list):
            raise ValueError(
                f"`configs` must be a `list`, but got {type(configs)}."
            )
        
        configs_ = copy.deepcopy(configs)
        objs     = {} if to_dict else []
        for config in configs_:
            if isinstance(config, str):
                name = config
                arch = None
            elif isinstance(config, dict):
                name = config.pop("name", None)
                arch = config.pop("arch", None)
                # kwargs |= config
            else:
                raise ValueError(f"Item inside `configs` must be a `str` or "
                                 f"`dict`, but got {type(config)}.")
            
            obj = self.build(name=name, arch=arch, to_dict=to_dict, **config)
            if obj:
                if to_dict:
                    objs |= obj
                else:
                    objs.append(obj)
        
        return objs if len(objs) > 0 else None

# endregion
