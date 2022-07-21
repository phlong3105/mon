#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base factory class for creating and registering classes.
"""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy

from torch import nn
from torch.optim import Optimizer
# noinspection PyUnresolvedReferences
from torch.optim.lr_scheduler import _LRScheduler

from one.core import Dict
from one.core import to_list
from one.core.types import assert_class
from one.core.types import assert_dict
from one.core.types import assert_dict_contain_key
from one.core.types import assert_list_of
from one.core.types import assert_same_length
from one.core.types import Callable
from one.core.types import is_list_of


# MARK: - Modules

class Registry:
	"""
	Base registry class for registering classes.

	Args:
		name (str):
			Registry name.
	"""
	
	# MARK: Magic Functions
	
	def __init__(self, name: str):
		self._name     = name
		self._registry = {}
	
	def __len__(self) -> int:
		"""
		The function returns the length of the registry.
		
		Returns:
		    The length of the registry.
		"""
		return len(self._registry)
	
	def __contains__(self, key: str):
		"""
		If the key is in the dictionary, return the value associated with the
		key, otherwise return the default value.
		
		Args:
		    key (str): The key to look for.
		
		Returns:
		    The value of the key.
		"""
		return self.get(key) is not None
	
	def __repr__(self):
		"""
		The __repr__ function returns a string that contains the name of the 
		class, the name of the object, and the contents of the registry.
		
		Returns:
		    The name of the class and the name of the registry.
		"""
		format_str = self.__class__.__name__ \
					 + f"(name={self._name}, items={self._registry})"
		return format_str
	
	# MARK: Properties
	
	@property
	def name(self) -> str:
		"""
		It returns the name of the object.
		
		Returns:
		    The name of the registry.
		"""
		return self._name
	
	@property
	def registry(self) -> dict:
		"""
		It returns the dictionary of the class.
		
		Returns:
		    A dictionary.
		"""
		return self._registry
	
	def get(self, key: str) -> Callable:
		"""
		If the key is in the registry, return the value.
		
		Args:
		    key (str): The name of the command.
		
		Returns:
		    A callable object.
		"""
		if key in self._registry:
			return self._registry[key]
	
	# MARK: Register
	
	def register(
		self,
		name  : str | None = None,
		module: Callable   = None,
		force : bool       = False
	) -> callable:
		"""
		It can be used as a normal method or as a decorator.
		
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
		    name (str | None): The name of the module. If not specified, it
		        will be the class name.
		    module (Callable): The module to register.
		    force (bool): If True, it will overwrite the existing module with
		        the same name. Defaults to False.
		
		Returns:
		    A function.
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
		module_class: Callable,
		module_name : str | None = None,
		force       : bool       = False
	):
		"""
		It takes a class and a name, and adds the class to the registry under
		the name.
		
		Args:
		    module_class (Callable): The class of the module to be registered.
		    module_name (str | None): The name of the module. If not provided,
		        it will be the class name in lowercase.
		    force (bool): If True, the module will be registered even if it's
		        already registered. Defaults to False.
		"""
		assert_class(module_class)
		
		if module_name is None:
			module_name = module_class.__name__.lower()
		module_name = to_list(module_name)
		
		for name in module_name:
			if not force and name in self._registry:
				continue
				# logger.debug(f"{name} is already registered in {self.name}.")
			else:
				self._registry[name] = module_class
	
	# MARK: Print

	def print(self):
		"""
		It prints the name of the object, and then prints the registry.
		"""
		from one.core.rich import console
		from one.core.rich import print_table
		
		console.log(f"[red]{self.name}:")
		print_table(self.registry)


class Factory(Registry):
	"""
	Default factory class for creating objects.
	
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
		"""
		It takes a name, and returns an instance of the class that is
		registered under that name.
		
		Args:
		    name (str): The name of the class to be built.
		
		Returns:
			An instance of the class that is registered with the name.
		"""
		assert_dict_contain_key(self.registry, name)
		instance = self.registry[name](*args, **kwargs)
		if not hasattr(instance, "name"):
			instance.name = name
		return instance
	
	def build_from_dict(self, cfg: Dict | None, **kwargs) -> object | None:
		"""
		Factory command to create a class' instance with arguments given in
		`cfgs`.
		
		Args:
			cfg (Dict | None): Class's arguments.
		
		Returns:
			Class's instance.
		"""
		if cfg is None:
			return None
		assert_dict(cfg)
		assert_dict_contain_key(cfg, "name")
	
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		cfgs: list[Dict] | None,
		**kwargs
	) -> list[object] | None:
		"""
		Factory command to create classes' instances with arguments given in
		`cfgs`.

		Args:
			cfgs (list[Dict] | None): List of classes' arguments.

		Returns:
			Classes' instances.
		"""
		if cfgs is None:
			return None
		assert_list_of(cfgs, dict)
		
		cfgs_     = deepcopy(cfgs)
		instances = []
		for cfg in cfgs_:
			name  = cfg.pop("name")
			cfg  |= kwargs
			instances.append(self.build(name=name, **cfg))
		
		return instances if len(instances) > 0 else None


class OptimizerFactory(Registry):
	"""
	Factory class for creating optimizers.
	"""
	
	# MARK: Build
	
	def build(
		self,
		net : nn.Module,
		name: str,
		*args, **kwargs
	) -> Optimizer | None:
		"""
		Factory command to create an optimizer with arguments given in
		`kwargs`.
		
		Args:
			net (nn.Module): Neural network module.
			name (str): Optimizer's name.
		
		Returns:
			Optimizer.
		"""
		assert_dict_contain_key(self.registry, name)
		return self.registry[name](params=net.parameters(), *args, **kwargs)
	
	def build_from_dict(
		self,
		net: nn.Module,
		cfg: Dict | None,
		**kwargs
	) -> Optimizer | None:
		"""
		Factory command to create an optimizer with arguments given in
		`cfgs`.

		Args:
			net (nn.Module): Neural network module.
			cfg (Dict | None): Optimizer's arguments.

		Returns:
			Optimizer.
		"""
		if cfg is None:
			return None
		assert_dict(cfg)
		assert_dict_contain_key(cfg, "name")
		
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(net=net, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		net : nn.Module,
		cfgs: list[Dict] | None,
		**kwargs
	) -> list[Optimizer] | None:
		"""
		Factory command to create optimizers with arguments given in `cfgs`.

		Args:
			net (nn.Module): List of neural network modules.
			cfgs (list[Dict] | None): List of optimizers' arguments.

		Returns:
			Optimizers.
		"""
		if cfgs is None:
			return None
		assert_list_of(cfgs, dict)
		
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
		cfgs: list[Dict] | None,
		**kwargs
	) -> list[Optimizer] | None:
		"""
		Factory command to create optimizers with arguments given in `cfgs`.

		Args:
			nets (list[nn.Module]): List of neural network modules.
			cfgs (list[Dict] | None): List of optimizers' arguments.

		Returns:
			Optimizers.
		"""
		if cfgs is None:
			return None
		assert_list_of(cfgs, dict)
		assert_list_of(nets, item_type=dict)
		assert_same_length(nets, cfgs)
	 
		cfgs_      = deepcopy(cfgs)
		optimizers = []
		for net, cfg in zip(nets, cfgs_):
			name  = cfg.pop("name")
			cfg  |= kwargs
			optimizers.append(self.build(net=net, name=name, **cfg))
		
		return optimizers if len(optimizers) > 0 else None


class SchedulerFactory(Registry):
	"""
	Factory class for creating schedulers.
	"""
	
	# MARK: Build
	
	def build(
		self,
		optimizer: Optimizer,
		name     : str | None,
		*args, **kwargs
	) -> _LRScheduler | None:
		"""
		Factory command to create a scheduler with arguments given in
		`kwargs`.
		
		Args:
			optimizer (Optimizer): Optimizer.
			name (str | None): Scheduler's name.
		
		Returns:
			Scheduler.
		"""
		if name is None:
			return None
		assert_dict_contain_key(self.registry, name)
	
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
		cfg      : Dict | None,
		**kwargs
	) -> _LRScheduler | None:
		"""
		Factory command to create a scheduler with arguments given in `cfg`.

		Args:
			optimizer (Optimizer): Optimizer.
			cfg (Dict | None): Scheduler's arguments.

		Returns:
			Scheduler.
		"""
		if cfg is None:
			return None
		assert_dict(cfg)
		assert_dict_contain_key(cfg, "name")
	
		cfg_  = deepcopy(cfg)
		name  = cfg_.pop("name")
		cfg_ |= kwargs
		return self.build(optimizer=optimizer, name=name, **cfg_)
	
	def build_from_dictlist(
		self,
		optimizer: Optimizer,
		cfgs     : Dict | None,
		**kwargs
	) -> list[_LRScheduler] | None:
		"""
		Factory command to create schedulers with arguments given in `cfgs`.

		Args:
			optimizer (Optimizer): Optimizer.
			cfgs (list[Dict] | None): List of schedulers' arguments.

		Returns:
			Schedulers.
		"""
		if cfgs is None:
			return None
		assert_list_of(cfgs, dict)
		
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
		cfgs      : list[list[Dict]] | None,
		**kwargs
	) -> list[_LRScheduler] | None:
		"""
		Factory command to create schedulers with arguments given in `cfgs`.

		Args:
			optimizers (list[Optimizer]): List of optimizers.
			cfgs (list[list[Dict]] | None): 2D-list of schedulers' arguments.

		Returns:
			Schedulers.
		"""
		if cfgs is None:
			return None
		if not (is_list_of(cfgs, item_type=list) or
		        all(is_list_of(cfg, item_type=dict) for cfg in cfgs)):
			raise TypeError(
				f"`cfgs` must be a 2D `list[dict]`. But got: {type(cfgs)}."
			)
		assert_same_length(optimizers, cfgs)
		
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


# MARK: - Factories
# NOTE: NN Layers
ACT_LAYERS           = Factory(name="act_layers")
ATTN_LAYERS          = Factory(name="attn_layers")
ATTN_POOL_LAYERS     = Factory(name="attn_pool_layers")
BOTTLENECK_LAYERS    = Factory(name="bottleneck_layers")
CONV_LAYERS          = Factory(name="conv_layers")
CONV_ACT_LAYERS      = Factory(name="conv_act_layers")
CONV_NORM_ACT_LAYERS = Factory(name="conv_norm_act_layers")
DROP_LAYERS          = Factory(name="drop_layers")
EMBED_LAYERS         = Factory(name="embed_layers")
HEADS 	             = Factory(name="heads")
LINEAR_LAYERS        = Factory(name="linear_layers")
MLP_LAYERS           = Factory(name="mlp_layers")
NORM_LAYERS          = Factory(name="norm_layers")
NORM_ACT_LAYERS      = Factory(name="norm_act_layers")
PADDING_LAYERS       = Factory(name="padding_layers")
PLUGIN_LAYERS        = Factory(name="plugin_layers")
POOL_LAYERS          = Factory(name="pool_layers")
RESIDUAL_BLOCKS      = Factory(name="residual_blocks")
SAMPLING_LAYERS      = Factory(name="sampling_layers")
# NOTE: Models
BACKBONES            = Factory(name="backbones")
CALLBACKS            = Factory(name="callbacks")
LOGGERS              = Factory(name="loggers")
LOSSES               = Factory(name="losses")
METRICS              = Factory(name="metrics")
MODELS               = Factory(name="models")
MODULE_WRAPPERS      = Factory(name="module_wrappers")
NECKS 	             = Factory(name="necks")
OPTIMIZERS           = OptimizerFactory(name="optimizers")
SCHEDULERS           = SchedulerFactory(name="schedulers")
# NOTE: Misc
AUGMENTS             = Factory(name="augments")
DATAMODULES          = Factory(name="datamodules")
DATASETS             = Factory(name="datasets")
DISTANCES            = Factory(name="distance_functions")
DISTANCE_FUNCS       = Factory(name="distance_functions")
FILE_HANDLERS        = Factory(name="file_handler")
LABEL_HANDLERS       = Factory(name="label_handlers")
MOTIONS              = Factory(name="motions")
TRANSFORMS           = Factory(name="transforms")


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
