#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base factory class for creating and registering classes.
"""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from typing import Union

from munch import Munch
from torch import nn
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler


# MARK: - Modules

class Registry:
	"""Base registry class for registering classes.

	Attributes:
		name (str):
			Registry name.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, name: str):
		self._name     = name
		self._registry = {}
	
	def __len__(self):
		return len(self._registry)
	
	def __contains__(self, key: str):
		return self.get(key) is not None
	
	def __repr__(self):
		format_str = self.__class__.__name__ \
					 + f"(name={self._name}, items={self._registry})"
		return format_str
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""Return the registry's name."""
		return self._name
	
	@property
	def registry(self) -> dict:
		"""Return the registry's dictionary."""
		return self._registry
	
	def get(self, key: str) -> "Callable":
		"""Get the registry record of the given `key`."""
		if key in self._registry:
			return self._registry[key]
	
	# MARK: Register
	
	def register(
		self,
		name  : Union[str, None] = None,
		module: "Callable"	     = None,
		force : bool             = False
	) -> callable:
		"""Register a module.

		A record will be added to `self._registry`, whose key is the class name
		or the specified name, and value is the class itself. It can be used
		as a decorator or a normal function.

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

		Args:
			name (str, None):
				Module name to be registered. If not specified, the class
				name will be used.
			module (type):
				Module class to be registered.
			force (bool):
				Whether to override an existing class with the same name.
		"""
		if not (name is None or isinstance(name, str)):
			raise TypeError(
				f"`name` must be `None` or `str`. But got: {type(name)}."
			)
		
		# NOTE: Use it as a normal method: x.register(module=SomeClass)
		if module is not None:
			self.register_module(module, name, force)
			return module
		
		# NOTE: Use it as a decorator: @x.register()
		def _register(cls):
			self.register_module(cls, name, force)
			return cls
		
		return _register
	
	def register_module(
		self,
		module_class: "Callable",
		module_name : Union[str, None] = None,
		force	    : bool 			   = False
	):
		if not inspect.isclass(module_class):
			raise TypeError(
				f"`module_class` must be a class type. "
				f"But got: {type(module_class)}."
			)
		
		if module_name is None:
			module_name = module_class.__name__.lower()
		
		if isinstance(module_name, str):
			module_name = [module_name]
		
		for name in module_name:
			if not force and name in self._registry:
				continue
				# logger.debug(f"{name} is already registered in {self.name}.")
			else:
				self._registry[name] = module_class
	
	# MARK: Print

	def print(self):
		"""Print the registry dictionary."""
		from one.core.rich import console
		from one.core.rich import print_table
		
		console.log(f"[red]{self.name}:")
		print_table(self.registry)


class Factory(Registry):
	"""Default factory class for creating objects.
	
	Registered object could be built from registry.
    Example:
        >>> MODELS = Factory("models")
        >>> @MODELS.register()
        >>> class ResNet:
        >>>     pass
        >>>
        >>> resnet_hparams = {}
        >>> resnet         = MODELS.build(name="ResNet", **resnet_hparams)
	"""
	
	# MARK: Build
	
	def build(self, name: str, *args, **kwargs) -> object:
		"""Factory command to create a class' instance with arguments given in
		`kwargs`.
		
		Args:
			name (str):
				Class's name.
			
		Returns:
			instance (object, None):
				Class' instance.
		"""
		if name not in self.registry:
			raise ValueError(f"`{name}` does not exist in the registry.")
		
		instance = self.registry[name](*args, **kwargs)
		if not hasattr(instance, "name"):
			instance.name = name
		return instance
	
	def build_from_dict(
		self, cfg: Union[dict, Munch, None], **kwargs
	) -> Union[object, None]:
		"""Factory command to create a class' instance with arguments given in
		`cfgs`.
		
		Args:
			cfg (dict, Munch, None):
				Class's arguments.
		
		Returns:
			instance (object, None):
				Class's instance.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			raise TypeError(f"`cfg` must be a `dict`.")
		if "name" not in cfg:
			raise ValueError(f"`cfg` dict must contain the key `name`.")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		cfgs: Union[list[Union[dict, Munch]], None],
		**kwargs
	) -> Union[list[object], None]:
		"""Factory command to create classes' instances with arguments given in
		`cfgs`.

		Args:
			cfgs (list[dict, Munch], None):
				List of classes' arguments.

		Returns:
			instances (list[object], None):
				Classes' instances.
		"""
		from .collection import is_list_of
		
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=dict) or
		        is_list_of(cfgs, item_type=Munch)):
			raise TypeError(f"`cfgs` must be a `list[dict]`.")
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(name=name, **cfg))
		
		return instances if len(instances) > 0 else None


class OptimizerFactory(Registry):
	"""Factory class for creating optimizers."""
	
	# MARK: Build
	
	def build(
		self,
		net : nn.Module,
		name: str,
		*args, **kwargs
	) -> Union[Optimizer, None]:
		"""Factory command to create an optimizer with arguments given in
		`kwargs`.
		
		Args:
			net (nn.Module):
				Neural network module.
			name (str):
				Optimizer's name.
		
		Returns:
			optimizer (Optimizer, None):
				Optimizer.
		"""
		if name not in self.registry:
			raise ValueError(f"{name} does not exist in the registry.")
		
		return self.registry[name](params=net.parameters(), *args, **kwargs)
	
	def build_from_dict(
		self,
		net: nn.Module,
		cfg: Union[Union[dict, Munch], None],
		**kwargs
	) -> Union[Optimizer, None]:
		"""Factory command to create an optimizer with arguments given in
		`cfgs`.

		Args:
			net (nn.Module):
				Neural network module.
			cfg (dict, Munch, None):
				Optimizer's arguments.

		Returns:
			optimizer (Optimizer, None):
				Optimizer.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			raise TypeError(f"`cfg` must be a `dict`.")
		if "name" not in cfg:
			raise ValueError(f"`cfg` dict must contain the key `name`.")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(net=net, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		net : nn.Module,
		cfgs: Union[list[Union[dict, Munch]], None],
		**kwargs
	) -> Union[list[Optimizer], None]:
		"""Factory command to create optimizers with arguments given in `cfgs`.

		Args:
			net (nn.Module):
				List of neural network modules.
			cfgs (list[dict, Munch], None):
				List of optimizers' arguments.

		Returns:
			optimizers (list[Optimizer], None):
				Optimizers.
		"""
		from one.core.collection import is_list_of
		
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=dict) or
		        is_list_of(cfgs, item_type=Munch)):
			raise TypeError(f"`cfgs` must be a `list[dict]`.")
		
		cfgs_      = deepcopy(cfgs)
		optimizers = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			optimizers.append(self.build(net=net, name=name, **cfg))
		
		return optimizers if len(optimizers) > 0 else None
	
	def build_from_list(
		self,
		nets: list[nn.Module],
		cfgs: Union[list[Union[dict, Munch]], None],
		**kwargs
	) -> Union[list[Optimizer], None]:
		"""Factory command to create optimizers with arguments given in `cfgs`.

		Args:
			nets (list[nn.Module]):
				List of neural network modules.
			cfgs (list[dict, Munch]):
				List of optimizers' arguments.

		Returns:
			optimizers (list[Optimizer], None):
				Optimizers.
		"""
		from one.core.collection import is_list_of
		
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=dict) or
		        is_list_of(cfgs, item_type=Munch)):
			raise TypeError(f"`cfgs` must be a `list[dict]`.")
		if not is_list_of(nets, item_type=dict):
			raise TypeError(
				f"`nets` must be a `list[nn.Module]`. But got: {nets}."
			)
		if len(nets) != len(cfgs):
			raise ValueError(
				f"`nets` and `cfgs` must have the same length. "
				f" But got: {len(nets)} != {len(cfgs)}."
			)
		
		cfgs_      = deepcopy(cfgs)
		optimizers = []
		for net, cfg in zip(nets, cfgs_):
			name  = cfg.pop("name")
			cfg  |= kwargs
			optimizers.append(self.build(net=net, name=name, **cfg))
		
		return optimizers if len(optimizers) > 0 else None


class SchedulerFactory(Registry):
	"""Factory class for creating schedulers."""
	
	# MARK: Build
	
	def build(
		self,
		optimizer: Optimizer,
		name     : Union[str, None],
		*args, **kwargs
	) -> Union[_LRScheduler, None]:
		"""Factory command to create a scheduler with arguments given in
		`kwargs`.
		
		Args:
			optimizer (Optimizer):
				Optimizer.
			name (str, None):
				Scheduler's name.
		
		Returns:
			scheduler (_LRScheduler, None):
				Scheduler.
		"""
		if name is None:
			return None
		if name not in self.registry:
			raise ValueError(f"{name} does not exist in the registry.")
		
		if name in ["gradual_warmup_scheduler"]:
			after_scheduler = kwargs.pop("after_scheduler")
			if isinstance(after_scheduler, dict):
				name_ = after_scheduler.pop("name")
				if name_ in self.registry:
					after_scheduler = self.registry[name_](
						optimizer=optimizer, **after_scheduler
					)
				else:
					after_scheduler = None
			return self.registry[name](
				optimizer       = optimizer,
				after_scheduler = after_scheduler,
				*args, **kwargs
			)
		
		return self.registry[name](optimizer=optimizer, *args, **kwargs)
	
	def build_from_dict(
		self,
		optimizer: Optimizer,
		cfg      : Union[Union[dict, Munch], None],
		**kwargs
	) -> Union[_LRScheduler, None]:
		"""Factory command to create a scheduler with arguments given in `cfg`.

		Args:
			optimizer (Optimizer):
				Optimizer.
			cfg (dict, Munch, None):
				Scheduler's arguments.

		Returns:
			scheduler (_LRScheduler, None):
				Scheduler.
		"""
		if cfg is None:
			return None
		if not isinstance(cfg, (dict, Munch)):
			raise TypeError(f"`cfg` must be a `dict`.")
		if "name" not in cfg:
			raise ValueError(f"`cfg` dict must contain the key `name`.")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(optimizer=optimizer, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		optimizer: Optimizer,
		cfgs     : Union[list[Union[dict, Munch]], None],
		**kwargs
	) -> Union[list[_LRScheduler], None]:
		"""Factory command to create schedulers with arguments given in `cfgs`.

		Args:
			optimizer (Optimizer):
				Optimizer.
			cfgs (list[dict, Munch], None):
				List of schedulers' arguments.

		Returns:
			schedulers (list[Optimizer], None):
				Schedulers.
		"""
		from one.core.collection import is_list_of
		
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=dict) or
		        is_list_of(cfgs, item_type=Munch)):
			raise TypeError(f"`cfgs` must be a `list[dict]`.")
		
		cfgs_      = deepcopy(cfgs)
		schedulers = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			schedulers.append(self.build(optimizer=optimizer, name=name, **cfg))
		
		return schedulers if len(schedulers) > 0 else None
	
	def build_from_list(
		self,
		optimizers: list[Optimizer],
		cfgs      : Union[list[list[Union[dict, Munch]]], None],
		**kwargs
	) -> Union[list[_LRScheduler], None]:
		"""Factory command to create schedulers with arguments given in `cfgs`.

		Args:
			optimizers (list[Optimizer]):
				List of optimizers.
			cfgs (list[list[dict, Munch]], None):
				2D-list of schedulers' arguments.

		Returns:
			schedulers (list[Optimizer], None):
				Schedulers.
		"""
		from one.core.collection import is_list_of
		
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=list) or
		        all(is_list_of(cfg, item_type=dict) for cfg in cfgs)):
			raise TypeError(
				f"`cfgs` must be a 2D `list[dict]`. But got: {type(cfgs)}."
			)
		if len(optimizers) != len(cfgs):
			raise ValueError(
				f"`optimizers` and `cfgs` must have the same length. "
				f"But got: {len(optimizers)} != {len(cfgs)}."
			)
			
		cfgs_      = deepcopy(cfgs)
		schedulers = []
		for optimizer, cfgs in zip(optimizers, cfgs_):
			for cfg in cfgs:
				name  = cfg.pop("name")
				cfg  |= kwargs
				schedulers.append(
					self.build(optimizer=optimizer, name=name, **cfg)
				)
		
		return schedulers if len(schedulers) > 0 else None


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
